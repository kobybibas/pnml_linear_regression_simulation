import logging
import time

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


def fit_least_squares_with_max_norm_constrain(phi_arr: np.ndarray, y: np.ndarray, max_norm: float) -> np.ndarray:
    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param phi_arr: data matrix. Each row represents an example.
    :param y: label vector
    :param max_norm: the constraint of the fitted parameters.
    :return: fitted parameters that satisfy the max norm constrain
    """
    n, m = phi_arr.shape
    assert n == len(y)

    theta_fit = fit_norm_constrained_least_squares_with_binary_search(phi_arr, y, max_norm)
    return theta_fit
    scale = 1e3

    def calc_theta_norm_based_on_lamb(lamb_root_fit):
        lamb_fit = lamb_root_fit ** 2  # to enforce positive lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        return calc_theta_norm(theta_fit)

    def mse(lamb_root_fit):
        lamb_fit = lamb_root_fit ** 2  # 0 <= lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        y_hat = (phi_arr @ theta_fit).squeeze()
        return npl.norm(y - y_hat) ** 2

    def calc_theta_derivatives(lamb_fit, theta_fit):
        phi_t_phi = phi_arr.T @ phi_arr
        inv = npl.pinv(phi_t_phi + lamb_fit * np.eye(phi_t_phi.shape[0], phi_t_phi.shape[1]))

        # scaling
        # inv *= scale

        derivative_first = inv @ theta_fit  # first derivative

        derivative_second = inv @ inv @ theta_fit + inv @ derivative_first
        return derivative_first, derivative_second

    def calc_mse_jac(lamb_root_fit):
        # derivative: dmse/dlamb
        lamb_fit = lamb_root_fit ** 2  # 0 <= lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)

        derivative_first, derivative_second = calc_theta_derivatives(lamb_fit, theta_fit)

        y_hat = (phi_arr @ theta_fit).squeeze()
        jac = np.mean(2 * (y - y_hat) * (phi_arr @ derivative_first).squeeze())
        return jac

    def calc_mse_hess(lamb_root_fit):
        # derivative: dmse/dlamb
        lamb_fit = lamb_root_fit ** 2  # 0 <= lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)

        derivative_first, derivative_second = calc_theta_derivatives(lamb_fit, theta_fit)
        y_hat = (phi_arr @ theta_fit).squeeze()

        first_term = np.mean(2 * (y - y_hat) * ((phi_arr @ derivative_first).squeeze()) ** 2)
        second_term = np.mean(2 * (y - y_hat) * (phi_arr @ derivative_second).squeeze())
        hess = first_term + second_term
        return np.array([[hess]])

    def calc_constrain_jac(lamb_root_fit):
        # derivative: dmse/dlamb
        lamb_fit = lamb_root_fit ** 2  # 0 <= lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        derivative_first, derivative_second = calc_theta_derivatives(lamb_fit, theta_fit)

        jac = 2 * theta_fit.T @ derivative_first
        return jac

    def calc_constrain_hess(lamb_root_fit, v):
        # derivative: dmse/dlamb
        lamb_fit = lamb_root_fit ** 2  # 0 <= lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        derivative_first, derivative_second = calc_theta_derivatives(lamb_fit, theta_fit)

        hess = 2 * derivative_first.T @ derivative_first + theta_fit.T @ derivative_second
        return hess * v

    # Optimize
    lamb_0 = choose_initial_guess(phi_arr, y, max_norm)
    inital_norm = calc_theta_norm_based_on_lamb(np.sqrt(lamb_0))
    non_linear_constraint = optimize.NonlinearConstraint(calc_theta_norm_based_on_lamb,
                                                         inital_norm, max_norm + np.finfo('float').eps,  # (min,max)
                                                         jac=calc_constrain_jac,
                                                         hess=calc_constrain_hess)
    res = optimize.minimize(mse, lamb_0,
                            method='trust-constr',  # method='SLSQP',
                            jac=calc_mse_jac,
                            hess=calc_mse_hess,
                            options=minimize_dict,
                            constraints=[non_linear_constraint])  # |theta|_2 <= max_norm

    # Verify output
    lamb_root = res.x
    lamb = float(lamb_root ** 2)
    theta = fit_least_squares_estimator(phi_arr, y, lamb=lamb)

    # Verify min norm solution
    norm = calc_theta_norm(theta)
    is_success = False
    if norm < max_norm and np.abs(max_norm - norm) < 0.1 and bool(res.success) is True:
        is_success = True

    logger.warning('fit_least_squares_with_max_norm_constrain:' + \
                   'phi_arr.shape={}. [|theta| max_norm]=[{:.5f} {:.5f}].'.format(phi_arr.shape,
                                                                                  calc_theta_norm(theta),
                                                                                  max_norm) + \
                   'lamb [initial fitted]=[{} {}'.format(lamb_0, lamb))

    if is_success is False:  # and not res.message == 'Positive directional derivative for linesearch':
        logger.warning(
            'fit_least_squares_with_max_norm_constrain:' + \
            'phi_arr.shape={}. [|theta| max_norm]=[{:.5f} {:.5f}].'.format(phi_arr.shape,
                                                                           calc_theta_norm(theta),
                                                                           max_norm) + \
            'lamb [initial fitted]=[{} {}'.format(lamb_0, lamb))
        logger.warning(res)

    return theta


def lamb_binary_search(phi_arr, y, max_norm, start: float = 0.0, end: float = 1e21,
                       tol_lamb: float = 1e-9, tol_func: float = 1e-6,
                       max_iter: int = 10000, log_space_avg_itr: int = 10) -> (float, dict):
    i = 0
    norm_start = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=start))
    norm_end = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=end))

    res_dict = {'success': False,
                'norm_start': [norm_start],
                'norm_end': [norm_end],
                'max_norm': max_norm,
                'start': [start],
                'end': [end],
                'message': '',
                'iter': i}

    # Check boundaries
    if np.abs(norm_start - max_norm) < tol_func:
        res_dict['success'] = True
        return start, res_dict
    if np.abs(norm_end - max_norm) < tol_lamb:
        res_dict['success'] = True
        return end, res_dict

    while True:

        # Update the middle lambda norm
        middle = 0.5 * (start + end) if i > log_space_avg_itr else 0.5 * (start + end / 10)
        norm_middle = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, lamb=middle))

        # If interval is too small: end search
        if np.abs(norm_start - norm_end) < tol_func:
            res_dict['success'] = True
            break
        if np.abs(start - end) < tol_func:
            res_dict['success'] = True
            break

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
        res_dict['norm_start'].append(norm_start)
        res_dict['norm_end'].append(norm_end)
        res_dict['start'].append(start)
        res_dict['end'].append(end)
        res_dict['iter'] = i
        i += 1

    return end, res_dict


def fit_norm_constrained_least_squares_with_binary_search(phi_arr: np.ndarray, y: np.ndarray,
                                                          max_norm: float) -> np.ndarray:
    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param phi_arr: data matrix. Each row represents an example.
    :param y: label vector
    :param max_norm: the constraint of the fitted parameters.
    :return: fitted parameters that satisfy the max norm constrain
    """
    t0 = time.time()
    n, m = phi_arr.shape
    assert n == len(y)
    tol_func, tol_lamb, max_iter = 1e-6, 1e-9, 1e4

    # Find lambda that produces norm that is lower than the max norm
    end, norm = 0.1, np.inf
    while norm > max_norm:
        end *= 10
        norm = calc_theta_norm(fit_least_squares_estimator(phi_arr, y, end))

    # Find the lambda that satisfies the max norm constrain
    lamb_fit, res_dict = lamb_binary_search(phi_arr, y, max_norm, start=0.0, end=end, tol_func=tol_func,
                                            tol_lamb=tol_lamb)

    # Produce theta
    theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
    norm = calc_theta_norm(theta_fit)
    mse = np.mean(calc_mse(phi_arr, y, theta_fit))

    # Verify results
    debug_str = f'[norm max_norm diff]=[{norm} {max_norm} {norm- max_norm}]]. mse={mse}.'
    if not norm < max_norm + tol_func or not np.abs(max_norm - norm) < 1e-1:
        logger.warning(debug_str)
    if res_dict['success'] is False:
        iter_num, start, end = res_dict['iter'], res_dict['start'][-1], res_dict['end'][-1]
        norm_start, norm_end, max_norm = res_dict['norm_start'][-1], res_dict['norm_end'][-1], res_dict['max_norm']
        debug_str_0 = f'phi_arr.shape={phi_arr.shape}. iter={iter_num} [start end[=[{start} {end}] ' + \
                      f'[norm_start norm_end max_norm]=[{norm_start} {norm_end} {max_norm}]'
        debug_str_1 = f'max_norm-[norm_start norm_end]=[{max_norm-norm_start} {max_norm-norm_end}]'
        debug_str_2 = f'lamb [start end]=[{start} {end}]'
        logger.warning('Optimization failed')
        logger.warning('    ' + res_dict['message'])
        logger.warning('    ' + debug_str_0)
        logger.warning('    ' + debug_str_1)
        logger.warning('    ' + debug_str_2)

    if False:
        iter_num = res_dict['iter']
        logger.warning(f'Finish in {iter_num} iterations. {time.time() - t0} sec')
    return theta_fit
