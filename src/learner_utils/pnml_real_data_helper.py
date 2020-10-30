import logging
import time

import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_logloss, calc_mse, calc_theta_norm, fit_least_squares_estimator
from learner_utils.optimization_utils import fit_norm_constrained_least_squares
from learner_utils.pnml_utils import Pnml, PnmlMinNorm, compute_pnml_logloss, add_test_to_train


def calc_empirical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                                    x_test: np.ndarray, y_test: np.ndarray,
                                    pnml_lambda_optim_dict: dict) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    n_train, n_features = x_train.shape

    # Initialize pNML
    n_train_effective = n_train + 1  # We add the test sample data to training
    if n_train_effective > n_features:
        pnml_h = Pnml(phi_train=x_train, y_train=y_train, lamb=0.0)
    else:
        pnml_h = PnmlMinNorm(constrain_factor=1.0, pnml_lambda_optim_dict=pnml_lambda_optim_dict,
                             phi_train=x_train, y_train=y_train, lamb=0.0)

    # Compute best variance using validation set
    best_vars = []
    for i, (x_i, y_i) in enumerate(zip(x_val, y_val)):
        t0 = time.time()
        best_var = pnml_h.optimize_variance(x_i, y_i)
        best_vars.append(best_var)

        # msg
        msg = '[{:03d}/{}] \t {:.1f} sec. x_train.shape={} \t best_var={}'.format(
            i, len(x_val) - 1, time.time() - t0, pnml_h.phi_train.shape, best_var)
        logger.info(msg)
    valset_mean_var = np.mean(best_vars)

    # Test set genies for logloss
    theta_erm = fit_least_squares_estimator(x_train, y_train, lamb=0.0)
    theta_test_genies = fit_genies_to_dataset(x_train, y_train, x_test, y_test, theta_erm)

    # Execute on test set
    nfs, success_list = [], []
    for j, x_j in enumerate(x_test):
        t0 = time.time()
        nf = pnml_h.calc_norm_factor(x_j, valset_mean_var)
        nfs.append(nf)

        # msg
        pnml_h_res_dict = pnml_h.res_dict
        message, success = pnml_h_res_dict['message'], pnml_h_res_dict['success']
        msg = '[{:03d}/{}] \t nf={:.8f}. \t {:.1f} sec. x_train.shape={} \t success={}. {}'.format(
            j, len(x_test) - 1, nf, time.time() - t0, pnml_h.phi_train.shape, success, message)
        success_list.append(success)
        if success is True:
            logger.info(msg)
        else:
            logger.warning(msg)

    regrets = np.log(nfs).tolist()
    logloss = compute_pnml_logloss(x_test, y_test, theta_test_genies, valset_mean_var, nfs)

    # Add to dict
    res_dict = {'empirical_pnml_regret': regrets,
                'empirical_pnml_test_logloss': logloss,
                'empirical_pnml_variance': [valset_mean_var] * len(x_test),
                'empirical_pnml_success': success_list}
    df = pd.DataFrame(res_dict)
    return df


def fit_genies_to_dataset(x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray, theta_erm: np.ndarray) -> list:
    n_train, num_features = x_train.shape

    theta_genies = []
    for x_test_i, y_test_i in zip(x_test, y_test):

        # Add test to train
        phi_arr, y = add_test_to_train(x_train, x_test_i), np.append(y_train, y_test_i)
        assert phi_arr.shape[0] == len(y)

        n_train_effective = n_train + 1  # We add the test sample data to training , therefore +1
        if n_train_effective > num_features:
            # Fit under param region
            theta_genie_i = fit_least_squares_estimator(phi_arr, y, lamb=0.0)
        else:
            # Fit over param region
            constrain = calc_theta_norm(theta_erm)
            theta_genie_i, lamb = fit_norm_constrained_least_squares(phi_arr, y, constrain)

        theta_genies.append(theta_genie_i)
    return theta_genies


def calc_genie_performance(x_train: np.ndarray, y_train: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray,
                           theta_erm: np.ndarray, variances: list) -> pd.DataFrame:
    # Fit genie to dataset
    theta_test_genies = fit_genies_to_dataset(x_train, y_train, x_test, y_test, theta_erm)

    # Metric
    test_logloss = [float(calc_logloss(x, y, theta_i, var_i)) for x, y, theta_i, var_i in
                    zip(x_test, y_test, theta_test_genies, variances)]
    test_mse = [float(calc_mse(x, y, theta_i)) for x, y, theta_i in zip(x_test, y_test, theta_test_genies)]
    theta_norm = [calc_theta_norm(theta_i) for theta_i in theta_test_genies]

    res_dict = {f'genie_test_mse': test_mse,
                f'genie_test_logloss': test_logloss,
                f'genie_theta_norm': theta_norm,
                f'genie_variance': variances}
    df = pd.DataFrame(res_dict)
    return df
