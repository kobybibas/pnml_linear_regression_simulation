import logging

import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_best_var
from learner_utils.learner_helpers import calc_square_error, calc_logloss, calc_theta_norm

logger_default = logging.getLogger(__name__)


def calc_theta_mn(x_arr: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    # x_arr: Each row is feature vec
    inv = np.linalg.pinv(x_arr.T @ x_arr)
    theta = inv @ x_arr.T @ y_vec
    return theta


def calc_mn_learner_performance(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                                x_test: np.ndarray, y_test: np.ndarray,
                                logger=logger_default) -> (pd.DataFrame, np.ndarray, float):
    theta_mn = calc_theta_mn(x_train, y_train)
    mn_valset_var = calc_best_var(x_val, y_val, theta_mn)

    res_dict_mn = {'mn_test_logloss': calc_logloss(x_test, y_test, theta_mn, mn_valset_var),
                   'mn_test_mse': calc_square_error(x_test, y_test, theta_mn),
                   'mn_theta_norm': calc_theta_norm(theta_mn),
                   'mn_variance': [mn_valset_var] * len(x_test)}
    df = pd.DataFrame(res_dict_mn)
    return df
