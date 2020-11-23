import logging

import numpy as np
import numpy.linalg as npl
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint

from learner_utils.learner_helpers import calc_theta_norm, fit_least_squares_estimator, calc_square_error


def calc_norm_with_svd(lamb: float, vt_y_square: np.ndarray, h_square: np.ndarray) -> float:
    norm = np.sum(vt_y_square[:len(h_square)] * h_square / (h_square + lamb) ** 2)
    return float(norm)


def check_tol_func(norm: float, norm_constraint: float, tol_func: float) -> bool:
    if np.abs(norm - norm_constraint) / norm_constraint < tol_func:
        return True
    else:
        return False


def execute_binary_lamb_search_svd(x_arr: np.ndarray, y_vec: np.ndarray, norm_constraint: float,
                                   lamb_left: float, lamb_right: float,
                                   tol_func: float = 1e-6, max_iter: int = 1e3,
                                   n_iter_for_scale: int = 10) -> (float, dict):
    # Calculate constants
    i, eps = 0, np.finfo('float').eps
    u, h, vt = npl.svd(x_arr.T, full_matrices=True)
    h_square = h ** 2
    vt_y_square = (vt @ y_vec) ** 2

    # Initialize edges
    norm_left = calc_theta_norm(fit_least_squares_estimator(x_arr, y_vec, lamb=lamb_left))
    norm_right = calc_norm_with_svd(lamb_right, vt_y_square, h_square)
    # norm_right = calc_theta_norm(fit_least_squares_estimator(x_arr, y_vec, lamb=lamb_right))

    # Initialize result dict for report
    optim_res_dict = {'success': False,
                      'norm_left': [norm_left],
                      'norm_right': [norm_right],
                      'norm_constraint': norm_constraint,
                      'start': [lamb_left],
                      'end': [lamb_right],
                      'message': '',
                      'iter': i}

    # Check boundaries
    if check_tol_func(norm_left, norm_constraint, tol_func):
        optim_res_dict['success'] = True
        return lamb_left, optim_res_dict
    elif check_tol_func(norm_right, norm_constraint, tol_func):
        optim_res_dict['success'] = True
        return lamb_right, optim_res_dict
    elif norm_left < norm_constraint:
        optim_res_dict['success'] = False
        optim_res_dict['message'] = 'Problem with left bound. '
        return lamb_left, optim_res_dict

    while True:
        # Update the middle lambda norm, first few iteration in more close to start
        middle = 0.5 * (lamb_left + lamb_right) if i > n_iter_for_scale else 0.5 * (lamb_left + lamb_right / 10)
        norm_middle = calc_norm_with_svd(middle, vt_y_square, h_square)
        # norm_middle= calc_theta_norm(fit_least_squares_estimator(x_arr, y_vec, lamb=middle))

        # Check if we can terminate
        if check_tol_func(norm_right, norm_constraint, tol_func):
            optim_res_dict['success'] = True
            optim_res_dict['message'] += 'tol_func was reached. '
            break
        elif i > max_iter:
            optim_res_dict['success'] = False
            optim_res_dict['message'] += 'Max iteration was reached. '

            # break the constraint if tolerance is too small
            if np.abs(lamb_right - lamb_left) < eps:
                lamb_right = lamb_left
                optim_res_dict['success'] = True
                optim_res_dict['message'] += 'tol lamb was reached. '
            break

        # Update edges
        if norm_middle > norm_constraint:
            lamb_left = middle
            norm_left = norm_middle
        else:
            lamb_right = middle
            norm_right = norm_middle

        # Update report dict
        optim_res_dict['norm_left'].append(norm_left)
        optim_res_dict['norm_right'].append(norm_right)
        optim_res_dict['start'].append(lamb_left)
        optim_res_dict['end'].append(lamb_right)
        optim_res_dict['iter'] = i
        i += 1

    return lamb_right, optim_res_dict


def find_upper_bound_lamb(phi_arr: np.ndarray, y: np.ndarray, max_norm: float) -> float:
    # Find lambda that produces norm that is lower than the max norm
    end, norm = 0.1, np.inf
    while norm > max_norm:
        end *= 10
        norm = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, end))
    return end


def print_msg(optim_res_dict: dict, phi_arr, norm, norm_constraint, mse, logger):
    iter_num, start, end = optim_res_dict['iter'], optim_res_dict['start'][-1], optim_res_dict['end'][-1]
    norm_left, norm_right = optim_res_dict['norm_left'][-1], optim_res_dict['norm_right'][-1]
    msg = 'Optimization failed: '
    msg += optim_res_dict['message'] + f' arr.shape={phi_arr.shape}. iter={iter_num} '
    msg += f' lambda [left right[=[{start} {end}] '
    msg += f'norm [left right constraint]=[{norm_left} {norm_right} {norm_constraint}] mse={mse}.'
    logger.warning(msg)


def fit_norm_constrained_least_squares(x_arr: np.ndarray, y_vec: np.ndarray, norm_constraint: float,
                                       tol_func: float = 1e-6, max_iter: int = 1e4,
                                       logger=logging.getLogger(__name__)) -> (np.ndarray, float, dict):
    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param x_arr: data matrix. Each row represents an example.
    :param y_vec: label vector
    :param norm_constraint: the constraint of the fitted parameters.
    :param tol_func: norm tolerance to terminate.
    :param max_iter: maximum optimization steps.
    :param logger: logger handler
    :return: fitted parameters that satisfy the max norm constrain
    """

    n, m = x_arr.shape
    assert n == len(y_vec)

    # After adding a sample, the norm must increase. If this is not the case, there is a stability issue, lamb ~=0
    lamb_fit = 0.0
    theta_fit = fit_least_squares_estimator(x_arr, y_vec, lamb=lamb_fit)
    if calc_theta_norm(theta_fit) <= norm_constraint + np.finfo('float').eps:
        optim_res_dict = {'fitted_lamb_list': lamb_fit, 'success': True}
        return theta_fit, lamb_fit, optim_res_dict

    # Initialize boundaries and find the lambda that satisfies the max norm constrain
    lamb_left, lamb_right = 0.0, find_upper_bound_lamb(x_arr, y_vec, norm_constraint)
    lamb_fit, optim_res_dict = execute_binary_lamb_search_svd(x_arr, y_vec, norm_constraint, lamb_left, lamb_right,
                                                              tol_func=tol_func, max_iter=max_iter)

    # Produce theta
    theta_fit = fit_least_squares_estimator(x_arr, y_vec, lamb=lamb_fit)
    norm = calc_theta_norm(theta_fit)
    mse = np.mean(calc_square_error(x_arr, y_vec, theta_fit))

    if optim_res_dict['success'] is False:
        print_msg(optim_res_dict, x_arr, norm, norm_constraint, mse, logger)
    return theta_fit, lamb_fit, optim_res_dict


def optimize_pnml_var(epsilon_square_gt: float, epsilon_square_list: list, y_trained_list: list,
                      pnml_var_optim_dict: dict) -> float:
    """
    Find the variance the minimizes the pNML loss
    :param epsilon_square_gt: \epsilon_gt^@ = (y_gt - x^\top \theta(z^N,x,y_gt))^2
    :param epsilon_square_list: \epsilon^@ = (y_trained - x^\top \theta(z^N,x,y_trained))^2
    :param y_trained_list: the labels that the genies where train
    :param pnml_var_optim_dict: optimization parameters
    :return: the optimal variance
    """
    epsilon_square_list = np.array(epsilon_square_list)
    y_trained_list = np.array(y_trained_list)
    eps = np.finfo('float').eps

    def genie_prob_edge(sigma_fit):
        genies_probs = calc_genie_probs(sigma_fit)
        return genies_probs[-1]

    def calc_genie_probs(sigma_fit):
        var_fit = sigma_fit ** 2
        genies_probs = np.exp(-epsilon_square_list / (2 * var_fit + eps)) / np.sqrt(2 * np.pi * var_fit + eps)
        return genies_probs

    def calc_nf(sigma_fit):
        # Genie probs
        genies_probs = calc_genie_probs(sigma_fit)

        # Normalization factor
        nf_fit = 2 * np.trapz(genies_probs, x=y_trained_list)
        return nf_fit

    def calc_jac(sigma_fit):
        var_fit = sigma_fit ** 2
        nf_fit = calc_nf(sigma_fit)
        jac = (1 / (2 * nf_fit * var_fit ** 2 + eps)) * (var_fit - nf_fit * epsilon_square_gt)
        return jac

    def calc_loss(sigma_fit):
        var_fit = sigma_fit ** 2
        nf_fit = calc_nf(sigma_fit)
        loss = 0.5 * np.log(2 * np.pi * var_fit + eps) + epsilon_square_gt / (2 * var_fit + eps) + np.log(nf_fit + eps)
        return loss

    def is_valid_sigma(sigma_fit) -> bool:
        nf_fit = calc_nf(sigma_fit)
        genie_edge_fit = genie_prob_edge(sigma_fit)

        is_valid = False
        if nf_fit > 1 and genie_edge_fit < eps:
            is_valid = True

        return is_valid

    # Optimize
    msgs_to_ignore = ['Positive directional derivative for linesearch',
                      'Iteration limit reached']
    sigma_best, loss_best = np.inf, np.inf
    sigma_0_interval = np.linspace(pnml_var_optim_dict['sigma_interval_min'],
                                   pnml_var_optim_dict['sigma_interval_max'],
                                   pnml_var_optim_dict['sigma_interval_steps'])
    debug_list = []
    for sigma_0 in sigma_0_interval:
        res = optimize.minimize(calc_loss, sigma_0, jac=calc_jac,
                                constraints=[NonlinearConstraint(calc_nf, 1.0, np.inf),
                                             NonlinearConstraint(genie_prob_edge, 0.0, eps)])
        if res.message in msgs_to_ignore:
            res.success = True
        if bool(res.success) is True and bool(res.fun < loss_best) and is_valid_sigma(res.x):
            loss_best = res.fun
            sigma_best = res.x
            sigma_0_best = sigma_0
        debug_list.append([{'sigma_0':sigma_0,
                            'res.message':res.message,
                            'res.x': res.x,
                            'nf': calc_nf(res.x),
                            'res.success':bool(res.success)
                            }])

    # Verify output
    var = float(sigma_best ** 2)
    return var
