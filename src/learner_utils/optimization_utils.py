import logging

import numpy as np
import numpy.linalg as npl
import scipy.optimize as optimize

from learner_utils.learner_helpers import calc_theta_norm, fit_least_squares_estimator, calc_mse

logger = logging.getLogger(__name__)

# Apply default minimization params
minimize_dict = {'disp': False, 'maxiter': 2000, 'xtol': 1e-12}


def choose_initial_guess(phi_arr, y, max_norm):
    lamb_chosen = 0.0
    lamb_list = [0.0, 1e-16, 1e-6, 1e-3, 1.0, 1e2, 1e3, 1e4, 1e6, 1e7, 1e8, 1e9]
    for lamb_fit in lamb_list:
        theta = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        norm = calc_theta_norm(theta)

        if norm <= max_norm:
            lamb_chosen = lamb_fit
            break

    return lamb_chosen




def execute_binary_lamb_search(phi_arr: np.ndarray, y: np.ndarray, max_norm: float, start: float = 0.0,
                               end: float = 1e21,
                               tol_lamb: float = 1e-9, tol_mse: float = 1e-6,
                               max_iter: int = 10000, n_iter_for_scale: int = 10) -> (float, dict):
    i = 0
    norm_start = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=start))
    norm_end = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=end))

    # Initialize result dict for report
    res_dict = {'success': False,
                'norm_start': [norm_start],
                'norm_end': [norm_end],
                'max_norm': max_norm,
                'start': [start],
                'end': [end],
                'message': '',
                'iter': i}

    # Check boundaries
    if np.abs(norm_start - max_norm) < tol_mse:
        res_dict['success'] = True
        return start, res_dict
    if np.abs(norm_end - max_norm) < tol_lamb:
        res_dict['success'] = True
        return end, res_dict

    while True:
        # Update the middle lambda norm, first few iteration in more close to start
        middle = 0.5 * (start + end) if i > n_iter_for_scale else 0.5 * (start + end / 10)
        norm_middle = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=middle))

        # If interval is too small: end search
        if np.abs(norm_start - norm_end) < tol_mse or np.abs(start - end) < tol_mse:
            res_dict['success'] = True
            break

        # Update edges
        if norm_middle > max_norm:
            start = middle
            norm_start = norm_middle
        else:
            end = middle
            norm_end = norm_middle

        # Verify interval is valid
        if i > max_iter:
            res_dict['message'] = 'Max iteration number succeed'
            break
        if norm_start < max_norm:
            res_dict['message'] = f'norm_start is smaller than the max norm: ' + \
                                  f'[norm_start max_norm diff]=[{norm_start} {max_norm} {norm_start-max_norm}]'
            break
        if norm_end > max_norm:
            res_dict['message'] = f'norm_end is greater than the max norm: ' + \
                                  f'[norm_end max_norm diff]=[{norm_end} {max_norm} {norm_end-max_norm}]'
            break

        # Update report dict
        res_dict['norm_start'].append(norm_start)
        res_dict['norm_end'].append(norm_end)
        res_dict['start'].append(start)
        res_dict['end'].append(end)
        res_dict['iter'] = i
        i += 1

    return end, res_dict


def fit_norm_constrained_least_squares(phi_arr: np.ndarray, y: np.ndarray, max_norm: float) -> np.ndarray:
    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param phi_arr: data matrix. Each row represents an example.
    :param y: label vector
    :param max_norm: the constraint of the fitted parameters.
    :return: fitted parameters that satisfy the max norm constrain
    """
    n, m = phi_arr.shape
    assert n == len(y)
    tol_func, tol_lamb, max_iter = 1e-6, 1e-9, 1e4

    # Find lambda that produces norm that is lower than the max norm
    end, norm = 0.1, np.inf
    while norm > max_norm:
        end *= 10
        norm = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, end))

    # Find the lambda that satisfies the max norm constrain
    lamb_fit, res_dict = execute_binary_lamb_search(phi_arr, y, max_norm, 0.0, end, tol_mse=tol_func, tol_lamb=tol_lamb)

    # Produce theta
    theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
    norm = calc_theta_norm(theta_fit)
    mse = np.mean(calc_mse(phi_arr, y, theta_fit))

    if res_dict['success'] is False or not norm < max_norm + tol_func or not np.abs(max_norm - norm) < 1e-1:
        iter_num, start, end = res_dict['iter'], res_dict['start'][-1], res_dict['end'][-1]
        norm_start, norm_end, max_norm = res_dict['norm_start'][-1], res_dict['norm_end'][-1], res_dict['max_norm']
        logger.warning('Optimization failed')
        logger.warning('    ' + res_dict['message'])
        logger.warning('    ' + f'phi_arr.shape={phi_arr.shape}. iter={iter_num} [start end[=[{start} {end}] ')
        logger.warning('    ' + f'[norm_start norm_end max_norm]=[{norm_start} {norm_end} {max_norm}]')
        logger.warning('    ' + f'max_norm-[norm_start norm_end]=[{max_norm-norm_start} {max_norm-norm_end}]')
        logger.warning('    ' + f'lamb [start end]=[{start} {end}]')
        logger.warning('    ' + f'[norm max_norm diff]=[{norm} {max_norm} {norm- max_norm}]]. mse={mse}.')
    return theta_fit
