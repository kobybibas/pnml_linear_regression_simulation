import logging
import time

import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_best_var, calc_var_with_valset
from learner_utils.learner_helpers import calc_logloss, calc_mse, calc_theta_norm, fit_least_squares_estimator
from learner_utils.optimization_utils import fit_norm_constrained_least_squares
from learner_utils.optimization_utils import optimize_pnml_var
from learner_utils.pnml_utils import Pnml, PnmlMinNorm, compute_pnml_logloss, add_test_to_train

logger = logging.getLogger(__name__)


def execute_pnml_on_sample(pnml_h: Pnml, x_i, y_i, theta_genie_i, var_dict, i: int, total: int):
    t0 = time.time()

    # Calc normalization factor
    y_vec = pnml_h.create_y_vec_to_eval(x_i)
    thetas = pnml_h.calc_genie_thetas(x_i, y_vec)

    # Calc best sigma
    epsilon_square_list = (y_vec - np.array([theta.T @ x_i for theta in thetas]).squeeze()) ** 2
    epsilon_square_true = (y_i - theta_genie_i.T @ x_i) ** 2
    best_var = optimize_pnml_var(epsilon_square_true, epsilon_square_list, y_vec)
    var_dict['pnml_adaptive_var'] = best_var

    # Calc genies predictions
    nf_dict = {}
    for var_name, var in var_dict.items():
        genies_probs, debug_dict = pnml_h.calc_probs_of_genies(x_i, y_vec, thetas, var)
        nf = 2 * np.trapz(genies_probs, x=y_vec)

        is_failed, message = verify_empirical_pnml_results(pnml_h, nf, pnml_h.phi_train, i)
        msg = '[{:03d}/{}] \t nf={:.8f}. \t {:.1f} sec. x_train.shape={} \t is_failed={}. {}={} {}'.format(
            i, total - 1, nf, time.time() - t0, pnml_h.phi_train.shape, is_failed, var_name, var, message)
        if True or is_failed is True:
            logger.warning(msg)
        else:
            logger.info(msg)
        nf_dict[var_name] = nf

    return nf_dict, best_var


def calc_empirical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray,
                                    x_val: np.ndarray, y_val: np.ndarray,
                                    x_test: np.ndarray, y_test: np.ndarray,
                                    theta_val_genies: np.ndarray, theta_test_genies: np.ndarray,
                                    genie_var_dict: dict) -> pd.DataFrame:
    n_train, n_features = x_train.shape

    # Initialize pNML
    n_train_effective = n_train + 1  # We add the test sample data to training
    if n_train_effective > n_features:
        pnml_h = Pnml(phi_train=x_train, y_train=y_train, lamb=0.0)
    else:
        pnml_h = PnmlMinNorm(constrain_factor=1.0, phi_train=x_train, y_train=y_train, lamb=0.0)

    # Compute best variance using validation set
    best_val_vars = []
    for i, (x_i, y_i, theta_genie_i) in enumerate(zip(x_val, y_val, theta_val_genies)):
        nf_dict_i, best_var = execute_pnml_on_sample(pnml_h, x_i, y_i, theta_genie_i, {}, i, len(y_val))
        best_val_vars.append(best_var)
    genie_var_dict['empirical_pnml_valset_mean_var'] = [float(np.mean(best_val_vars))] * len(x_test)

    # Execute on test set
    best_test_vars = []
    nf_dict = {var_type: [] for var_type in genie_var_dict.keys()}
    nf_dict.update({'empirical_pnml_adaptive_var': []})
    for i, (x_i, y_i, theta_genie_i) in enumerate(zip(x_test, y_test, theta_test_genies)):
        var_dict = {var_name: var_values[i] for var_name, var_values in genie_var_dict.items()}
        nf_dict_i, best_var = execute_pnml_on_sample(pnml_h, x_i, y_i, theta_genie_i, var_dict, i, len(y_test))
        for nf_name, nf_value in nf_dict_i.items():
            nf_dict[nf_name].append(nf_value)
        best_test_vars.append(best_var)
    genie_var_dict['empirical_pnml_adaptive_var'] = best_test_vars

    res_dict = {}
    for var_type in genie_var_dict.keys():
        nfs = nf_dict[var_type]
        regrets = np.log(nfs).tolist()
        genie_vars = genie_var_dict[var_type]
        logloss = compute_pnml_logloss(x_test, y_test, theta_test_genies, genie_vars, nfs)

        # Add to dict
        res_dict.update({f'empirical_pnml_{var_type}_regret': regrets,
                         f'empirical_pnml_{var_type}_test_logloss': logloss,
                         f'empirical_pnml_{var_type}_variance': genie_vars})
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
            theta_genie_i = fit_norm_constrained_least_squares(phi_arr, y, constrain)

        theta_genies.append(theta_genie_i)
    return theta_genies


def calc_genie_performance(x_train: np.ndarray, y_train: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray,
                           theta_erm: np.ndarray) -> (pd.DataFrame, list, dict):
    # Minimum norm
    theta_mn = fit_least_squares_estimator(x_train, y_train, lamb=0.0)
    mn_var = calc_var_with_valset(x_val, y_val, theta_mn)

    # Fit genie to dataset
    theta_test_genies = fit_genies_to_dataset(x_train, y_train, x_test, y_test, theta_erm)
    theta_val_genies = fit_genies_to_dataset(x_train, y_train, x_val, y_val, theta_erm)

    # Fit vats
    testset_adaptive_var = [calc_best_var(x_test_i, y_test_i, theta_genie_i) for x_test_i, y_test_i, theta_genie_i in
                            zip(x_test, y_test, theta_test_genies)]

    valset_adaptive_var = [calc_best_var(x_test_i, y_test_i, theta_genie_i) for x_test_i, y_test_i, theta_genie_i in
                           zip(x_val, y_val, theta_val_genies)]

    # Different vars to experiment
    n_test = len(x_test)
    var_dict = {'genie_adaptive_var': testset_adaptive_var,
                'mn_var': [mn_var] * n_test,
                'genie_valset_mean_var': [np.mean(valset_adaptive_var)] * n_test}

    # Metric
    res_dict = {}
    for var_name, var_values in var_dict.items():
        test_logloss = [float(calc_logloss(x, y, theta_i, var_i)) for x, y, theta_i, var_i in
                        zip(x_test, y_test, theta_test_genies, var_values)]
        test_mse = [float(calc_mse(x, y, theta_i)) for x, y, theta_i in zip(x_test, y_test, theta_test_genies)]
        theta_norm = [calc_theta_norm(theta_i) for theta_i in theta_test_genies]

        res_dict.update({f'genie_{var_name}_test_mse': test_mse,
                         f'genie_{var_name}_test_logloss': test_logloss,
                         f'genie_{var_name}_theta_norm': theta_norm,
                         f'genie_{var_name}_variance': var_values})
    df = pd.DataFrame(res_dict)

    return df, theta_test_genies, theta_val_genies, var_dict
