import logging

import numpy as np
import numpy.linalg as npl
import pandas as pd
import scipy.optimize

from learner_utils.learner_helpers import calc_mse, calc_logloss, calc_theta_norm, calc_best_var

logger = logging.getLogger(__name__)


def compute_practical_mdl_comp(x_train, y_train, variance: float = 1.0, x0: float = 1e-10):
    """
    Calculate prac-mdl-comp for this dataset
    :param x_train:
    :param y_train:
    :param variance:
    :param x0: initial gauss
    :return:
    """

    # My addition: npl.eig -> npl.eigh
    eigenvals, eigenvecs = npl.eigh(x_train.T @ x_train)

    def calc_theta_hat(l):
        inv = npl.pinv(x_train.T @ x_train + l * np.eye(x_train.shape[1]))
        return inv @ x_train.T @ y_train

    def prac_mdl_comp_objective(l):
        np.seterr(divide='raise')
        theta_hat = calc_theta_hat(l)
        mse_norm = npl.norm(y_train - x_train @ theta_hat) ** 2 / (2 * variance)
        theta_norm = npl.norm(theta_hat) ** 2 / (2 * variance)
        eigensum = 0.5 * np.sum(np.log((eigenvals + l) / l))
        return (mse_norm + theta_norm + eigensum) / y_train.size

    opt_solved = scipy.optimize.minimize(prac_mdl_comp_objective, x0=x0)
    prac_mdl = opt_solved.fun
    lambda_opt = opt_solved.x
    theta_hat = calc_theta_hat(lambda_opt)

    return {'prac_mdl': prac_mdl, 'lambda_opt': lambda_opt, 'theta_hat': theta_hat}


def calc_mdl_performance(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray, var: float) -> pd.DataFrame:
    with np.errstate(all='ignore'):
        mdl_dict = compute_practical_mdl_comp(x_train, y_train, variance=var)
    theta_mdl = mdl_dict['theta_hat']
    lambda_opt = mdl_dict['lambda_opt']
    var = calc_best_var(x_val, y_val, theta_mdl)

    n_test = len(x_test)
    res_dict = {'mdl_lambda_opt': [lambda_opt] * n_test,
                'mdl_test_mse': calc_mse(x_test, y_test, theta_mdl),
                'mdl_theta_norm': [calc_theta_norm(theta_mdl)] * n_test,
                'mdl_test_logloss': calc_logloss(x_test, y_test, theta_mdl, var),
                'mdl_variance': [var] * n_test}

    df = pd.DataFrame(res_dict)
    return df
