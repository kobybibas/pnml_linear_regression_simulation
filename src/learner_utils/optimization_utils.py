import logging

import numpy as np
import scipy.optimize as optimize

from learner_utils.learner_helpers import calc_theta_norm, fit_least_squares_estimator, calc_mse

logger = logging.getLogger(__name__)


def execute_binary_lamb_search(phi_arr: np.ndarray, y: np.ndarray, max_norm: float,
                               lamb_lower: float = 0.0, lamb_upper: float = 1e21,
                               tol_lamb: float = 1e-9, tol_func: float = 1e-6,
                               max_iter: int = 10000, n_iter_for_scale: int = 10) -> (float, dict):
    i = 0
    norm_start = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=lamb_lower))
    norm_end = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=lamb_upper))

    # Initialize result dict for report
    res_dict = {'success': False,
                'norm_start': [norm_start],
                'norm_end': [norm_end],
                'max_norm': max_norm,
                'start': [lamb_lower],
                'end': [lamb_upper],
                'message': '',
                'iter': i}

    # Check boundaries
    if norm_start < max_norm:
        res_dict['success'] = True
        return lamb_lower, res_dict
    if np.abs(norm_start - max_norm) < tol_func:
        res_dict['success'] = True
        return lamb_lower, res_dict
    if np.abs(norm_end - max_norm) < tol_func:
        res_dict['success'] = True
        return lamb_upper, res_dict

    while True:
        # Update the middle lambda norm, first few iteration in more close to start
        middle = 0.5 * (lamb_lower + lamb_upper) if i > n_iter_for_scale else 0.5 * (lamb_lower + lamb_upper / 10)
        norm_middle = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=middle))

        # If interval is too small: end search
        if np.abs(norm_start - norm_end) < tol_func:
            res_dict['success'] = True
            res_dict['message'] += 'tol_func was reached. '
            break
        if np.abs(lamb_lower - lamb_upper) < tol_lamb:
            res_dict['success'] = True
            res_dict['message'] += 'tol_lamb was reached. '
            break

        # Update edges
        if norm_middle > max_norm:
            lamb_lower = middle
            norm_start = norm_middle
        else:
            lamb_upper = middle
            norm_end = norm_middle

        # Verify interval is valid
        if i > max_iter:
            res_dict['message'] += 'Max iteration was reached. '
            break
        if norm_start < max_norm:
            res_dict['message'] += f'norm_start is smaller than the max norm: ' + \
                                   f'[norm_start max_norm diff]=[{norm_start} {max_norm} {norm_start-max_norm}] '
            break
        if norm_end > max_norm:
            res_dict['message'] += f'norm_end is greater than the max norm: ' + \
                                   f'[norm_end max_norm diff]=[{norm_end} {max_norm} {norm_end-max_norm}] '
            break

        # Update report dict
        res_dict['norm_start'].append(norm_start)
        res_dict['norm_end'].append(norm_end)
        res_dict['start'].append(lamb_lower)
        res_dict['end'].append(lamb_upper)
        res_dict['iter'] = i
        i += 1

    if norm_end > max_norm:
        res_dict['message'] += 'norm is greater than max norm.'
        res_dict['success'] = False
    if np.abs(max_norm - norm_end) > 0.01 * max_norm:
        res_dict['message'] += 'norm is far from the max norm by more the 1%.'
        res_dict['success'] = False

    return lamb_upper, res_dict


def find_upper_bound_lamb(phi_arr: np.ndarray, y: np.ndarray, max_norm: float) -> float:
    # Find lambda that produces norm that is lower than the max norm
    end, norm = 0.1, np.inf
    while norm > max_norm:
        end *= 10
        norm = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, end))
    return end


def fit_norm_constrained_least_squares(phi_arr: np.ndarray, y: np.ndarray, max_norm: float,
                                       tol_func: float = 1 - 6, tol_lamb: float = 1e-9,
                                       max_iter: int = 1e4) -> (np.ndarray, float):
    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param phi_arr: data matrix. Each row represents an example.
    :param y: label vector
    :param max_norm: the constraint of the fitted parameters.
    :return: fitted parameters that satisfy the max norm constrain
    """
    n, m = phi_arr.shape
    assert n == len(y)

    lamb_lower = 0.0
    lamb_upper = find_upper_bound_lamb(phi_arr, y, max_norm)

    # Find the lambda that satisfies the max norm constrain
    lamb_fit, res_dict = execute_binary_lamb_search(phi_arr, y, max_norm, lamb_lower, lamb_upper,
                                                    tol_func=tol_func, tol_lamb=tol_lamb, max_iter=max_iter)

    # Produce theta
    theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
    norm = calc_theta_norm(theta_fit)
    mse = np.mean(calc_mse(phi_arr, y, theta_fit))

    if res_dict['success'] is False:
        iter_num, start, end = res_dict['iter'], res_dict['start'][-1], res_dict['end'][-1]
        norm_start, norm_end, max_norm = res_dict['norm_start'][-1], res_dict['norm_end'][-1], res_dict['max_norm']
        logger.warning('Optimization failed')
        logger.warning('    ' + res_dict['message'])
        logger.warning('    ' + f'phi_arr.shape={phi_arr.shape}. iter={iter_num} [start end[=[{start} {end}] ')
        logger.warning('    ' + f'[norm_start norm_end max_norm]=[{norm_start} {norm_end} {max_norm}]')
        logger.warning('    ' + f'max_norm-[norm_start norm_end]=[{max_norm-norm_start} {max_norm-norm_end}]')
        logger.warning('    ' + f'lamb [start end]=[{start} {end}]')
        logger.warning('    ' + f'[norm max_norm diff]=[{norm} {max_norm} {norm- max_norm}]. mse={mse}.')
    return theta_fit, lamb_fit


def optimize_pnml_var(epsilon_square_gt: float, epsilon_square_list: list, y_trained_list: list) -> np.ndarray:
    epsilon_square_list = np.array(epsilon_square_list)
    y_trained_list = np.array(y_trained_list)

    def calc_nf(sigma_fit):
        var_fit = sigma_fit ** 2

        # Genie probs
        genies_probs = np.exp(-epsilon_square_list / (2 * var_fit)) / np.sqrt(2 * np.pi * var_fit)

        # Normalization factor
        nf = 2 * np.trapz(genies_probs, x=y_trained_list)
        return nf

    def calc_jac(sigma_fit):
        var_fit = sigma_fit ** 2
        nf = calc_nf(sigma_fit)

        jac = (1 / (2 * nf * var_fit ** 2)) * (var_fit - nf * epsilon_square_gt)
        return jac

    def calc_loss(sigma_fit):
        var_fit = sigma_fit ** 2

        # Genie probs
        nf = calc_nf(sigma_fit)

        loss = 0.5 * np.log(2 * np.pi * var_fit) + epsilon_square_gt / (2 * var_fit) + np.log(nf)
        return loss

    # Optimize
    sigma_0 = 0.001
    res = optimize.minimize(calc_loss, sigma_0, jac=calc_jac)

    # Verify output
    sigma = res.x
    var = float(sigma ** 2)

    if bool(res.success) is False and \
            not res.message == 'Desired error not necessarily achieved due to precision loss.':
        logger.warning(res.message)
    return var
