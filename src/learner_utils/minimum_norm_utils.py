import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_best_var
from learner_utils.learner_helpers import calc_mse, calc_logloss, calc_theta_norm
from learner_utils.learner_helpers import calc_var_with_valset
from learner_utils.pnml_utils import add_test_to_train


def calc_theta_mn(x_arr: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    # x_arr: Each row is feature vec
    inv = np.linalg.pinv(x_arr.T @ x_arr)
    theta = inv @ x_arr.T @ y_vec
    return theta


def calc_mn_learner_performance(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                                x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    theta_mn = calc_theta_mn(x_train, y_train)
    var = calc_var_with_valset(x_val, y_val, theta_mn)

    mn_test_logloss_adaptive_var, var_list = [], []
    for x_test_i, y_test_i, in zip(x_test, y_test):
        phi_arr, y = add_test_to_train(x_train, x_test_i), np.append(y_train, y_test_i)
        var_i = calc_best_var(phi_arr, y, theta_mn)

        # Save
        var_list.append(var_i)
        mn_test_logloss_adaptive_var.append(float(calc_logloss(x_test_i, y_test_i, theta_mn, var_i)))

    res_dict_mn_fixed_var = {'mn_test_logloss': calc_logloss(x_test, y_test, theta_mn, var),
                             'mn_test_mse': calc_mse(x_test, y_test, theta_mn),
                             'mn_theta_norm': calc_theta_norm(theta_mn),
                             'mn_variance': [var] * len(x_test)}

    res_dict_mn_adaptive_var = {'mn_adaptive_var_test_logloss': mn_test_logloss_adaptive_var,
                                'mn_adaptive_var_variance': var_list}

    res_dict_mn = {}
    res_dict_mn.update(res_dict_mn_fixed_var)
    res_dict_mn.update(res_dict_mn_adaptive_var)
    df = pd.DataFrame(res_dict_mn)
    return df
