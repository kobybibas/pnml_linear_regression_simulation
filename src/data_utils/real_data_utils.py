import logging
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pmlb
import ray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from learner_utils.analytical_pnml_utils import calc_analytical_pnml_performance, calc_genie_performance
from learner_utils.learner_helpers import estimate_base_var
from learner_utils.mdl_utils import calc_mdl_performance
from learner_utils.minimum_norm_utils import calc_mn_learner_performance, calc_theta_mn
from learner_utils.pnml_utils import calc_empirical_pnml_performance

logger = logging.getLogger(__name__)


def create_trainset_sizes_to_eval(n_train: int, n_features: int,
                                  num_trainset_sizes_over_param: int,
                                  is_under_param_region: bool, num_trainset_sizes_under_params: int) -> list:
    """
    Create list of training set sizes to evaluate.
    :param n_train: Training set size.
    :param n_features:  Number of feature
    :param num_trainset_sizes_over_param: How much training set sizes to evaluate in the overparam region.
    :param is_under_param_region: Whether to evaluate the under param region.
    :param num_trainset_sizes_under_params:  How much training set sizes to evaluate in the underparam region.
    :return: list of training set sizes.
    """
    # Define train set to evaluate
    trainset_sizes = np.unique(np.linspace(3, n_features, num_trainset_sizes_over_param).astype(int))
    if is_under_param_region is True:
        trainset_sizes_1 = np.linspace(n_features, n_train, num_trainset_sizes_under_params).round().astype(int)
        trainset_sizes = np.unique(np.append(trainset_sizes, trainset_sizes_1))
    return trainset_sizes


def standardize_samples(x_arr: np.ndarray, y: np.ndarray):
    # Normalize the data. To match mdl-comp preprocess
    x_mean, x_std = np.mean(x_arr, axis=1).reshape(-1, 1), np.std(x_arr, axis=1).reshape(-1, 1)
    x_std[x_std < 1e-12] = 1
    x_stand = x_arr - x_mean
    x_stand = x_stand / x_std
    y_stand = (y - np.mean(y)) / np.std(y)
    return x_stand, y_stand


def standardize_feature(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> (np.ndarray,
                                                                                        np.ndarray,
                                                                                        np.ndarray):
    x_train_stand = x_train.copy()
    x_val_stand = x_val.copy()
    x_test_stand = x_test.copy()

    for k in range(x_train.shape[1]):
        # Fit on training data columns
        scale = StandardScaler().fit(x_train_stand[:, k].reshape(-1, 1))

        # Transform the data column features (columns)
        x_train_stand[:, k] = scale.transform(x_train_stand[:, k].reshape(-1, 1)).squeeze()
        x_val_stand[:, k] = scale.transform(x_val_stand[:, k].reshape(-1, 1)).squeeze()
        x_test_stand[:, k] = scale.transform(x_test_stand[:, k].reshape(-1, 1)).squeeze()

    return x_train_stand, x_val_stand, x_test_stand


def get_data(dataset_name: str, data_dir: str = None, is_add_bias_term: bool = True):
    x_all, y_all = pmlb.fetch_data(dataset_name, return_X_y=True, local_cache_dir=data_dir)
    if is_add_bias_term is True:
        x_all = np.hstack((x_all, np.ones((x_all.shape[0], 1))))
    return x_all, y_all


def split_dataset(x_all, y_all, is_standardize_feature: bool = False, is_standardize_samples: bool = True):
    if is_standardize_samples is True:
        # Normalize the data. To match mdl-comp preprocess
        x_all, y_all = standardize_samples(x_all, y_all)

    # Split: train val test = [0.6, 0.2 ,0.2]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=True)

    if is_standardize_feature is True:
        # apply standardization on numerical features
        x_train, x_val, x_test = standardize_feature(x_train, x_val, x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def add_trail_to_res_dict(res_dict: dict, trail_dict: dict, method_name: str):
    # Add method to dict
    if method_name not in res_dict:
        res_dict[method_name] = defaultdict(list)

    # Add trail results
    for key, value in trail_dict.items():
        res_dict[method_name][key].append(value)


def fit_least_squares_with_lambda(phi_arr: np.ndarray, y: np.ndarray, phi_test, var,
                                  minimize_dict: dict = None) -> np.ndarray:
    from learner_utils.pnml_utils import fit_least_squares_estimator
    from learner_utils.analytical_pnml_utils import AnalyticalPNML
    import scipy.optimize as optimize

    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param phi_arr: data matrix. Each row represents an example.
    :param y: label vector
    :param max_norm: the constraint of the fitted parameters.
    :param minimize_dict: configuration for minimization function.
    :return: fitted parameters that satisfy the max norm constrain
    """
    n, m = phi_arr.shape
    assert n == len(y)

    theta_erm = fit_least_squares_estimator(phi_arr, y, lamb=0.0)
    pnml_h = AnalyticalPNML(phi_arr, theta_erm)

    def logloss_and_regret(lamb_root_fit):
        lamb_fit = lamb_root_fit ** 2  # 0 <= lambda
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        y_hat = (phi_arr @ theta_fit).squeeze()

        logloss = np.sum((1 / (2 * var)) * (y - y_hat) ** 2) + n * np.log(np.sqrt(2 * np.pi * var))
        nf = pnml_h.calc_norm_factor_with_lambda(phi_test, lamb_fit)
        return (1 / (n + 1)) * (logloss + np.log(nf))

    # Initial gauss
    lamb_0 = 1e-6

    # Optimize
    res = optimize.minimize(logloss_and_regret, lamb_0, method='SLSQP')
    # Verify output
    lamb = res.x ** 2
    theta = fit_least_squares_estimator(phi_arr, y, lamb=lamb)

    if bool(res.success) is False:
        logger.warning('fit_least_squares_with_max_norm_constrain: Failed')
        logger.warning('lamb [initial fitted]=[{} {}]'.format(lamb_0, lamb))
        logger.warning(res)
    return theta


def calc_lambda_pnml_performance(x_train, y_train, x_test, y_test, var):
    from learner_utils.learner_helpers import compute_logloss
    # Find lambda that minimize
    thetas, regrets, norm_factors = [], [], []
    for j, (x_test_i, y_test_i) in enumerate(zip(x_test, y_test)):
        t0 = time.time()

        theta = fit_least_squares_with_lambda(x_train, y_train, x_test_i, var)
        thetas.append(theta)

    res_dict = {'test_logloss': compute_logloss(x_test, y_test, theta, var) for theta in thetas}
    return res_dict


@ray.remote
def execute_trail(x_all: np.ndarray, y_all: np.ndarray, trail_num: int, trainset_size: int, dataset_name: str,
                  is_eval_mdl: bool, is_eval_empirical_pnml: bool, is_eval_analytical_pnml: bool,
                  is_eval_lambda_pnml: bool,
                  is_adaptive_var: bool,
                  is_standardize_feature: bool, is_standardize_samples: bool,
                  debug_print: bool = True):
    t0 = time.time()

    # Execute trails
    df_list = []

    # Split dataset
    t1 = time.time()
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_all, y_all,
                                                                   is_standardize_feature=is_standardize_feature,
                                                                   is_standardize_samples=is_standardize_samples)
    x_train, y_train = x_train[:trainset_size, :], y_train[:trainset_size]
    debug_print and logger.info('split_dataset in {:.3f} sec'.format(time.time() - t1))

    # General statistics
    n_test = len(x_test)
    param_df = pd.DataFrame({'dataset_name': [dataset_name] * n_test,
                             'trainset_size': [x_train.shape[0]] * n_test,
                             'trail_num': [trail_num] * n_test,
                             'valset_size': [x_val.shape[0]] * n_test,
                             'testset_size': [x_test.shape[0]] * n_test,
                             'num_features': [x_train.shape[1]] * n_test})
    df_list.append(param_df)

    # Compute variance
    theta_mn = calc_theta_mn(x_train, y_train)
    var = estimate_base_var(x_train, y_train, x_val, y_val, theta_mn, is_adaptive_var=is_adaptive_var)

    # Minimum norm learner
    t1 = time.time()
    mn_df = calc_mn_learner_performance(x_train, y_train, x_val, y_val, x_test, y_test, theta_mn=theta_mn)
    df_list.append(mn_df)
    debug_print and logger.info('calc_mn_learner_performance in {:.3f} sec'.format(time.time() - t1))

    # Genie
    t1 = time.time()
    theta_genies = []  # Fill this by the calc_empirical_pnml_performance func
    genie_df = calc_genie_performance(x_train, y_train, x_val, y_val, x_test, y_test, theta_mn, theta_genies)
    df_list.append(genie_df)
    theta_genies = theta_genies[0]
    debug_print and logger.info('calc_genie_performance in {:.3f} sec'.format(time.time() - t1))

    # Empirical pNML learner
    t1 = time.time()
    if is_eval_empirical_pnml is True:
        pnml_df = calc_empirical_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test,
                                                  theta_genies, theta_mn)
        df_list.append(pnml_df)
    debug_print and logger.info('calc_empirical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # Analytical pNML learner
    t1 = time.time()
    if is_eval_analytical_pnml is True:
        analytical_pnml_df, analytical_pnml_isit_df = calc_analytical_pnml_performance(x_train, y_train,
                                                                                       x_val, y_val,
                                                                                       x_test, y_test,
                                                                                       theta_genies, theta_mn)
        df_list += [analytical_pnml_df, analytical_pnml_isit_df]
    debug_print and logger.info('calc_analytical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # # Analytical pNML learner
    # t1 = time.time()
    # if is_eval_lambda_pnml is True:
    #     lambda_pnml = calc_lambda_pnml_performance(x_train, y_train, x_test, y_test, var)
    #     add_trail_to_res_dict(res_trails_dict, lambda_pnml, 'lambda_pnml')
    # debug_print and logger.info('calc_lambda_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # MDL
    t1 = time.time()
    if is_eval_mdl is True:
        mdl_df = calc_mdl_performance(x_train, y_train, x_val, y_val, x_test, y_test, var)
        df_list.append(mdl_df)
    debug_print and logger.info('calc_mdl_performance in {:.3f} sec'.format(time.time() - t1))

    res_df = pd.concat(df_list, axis=1, sort=False)
    res_df['test_idx'] = res_df.index
    debug_print and logger.info(
        '    Finish {} trainset_size: {} in {:.3f}'.format(dataset_name, trainset_size, time.time() - t0))
    return res_df


def download_regression_datasets(data_dir: str, out_path: str):
    t0 = time.time()
    datasets_dict = {}
    for dataset_name in tqdm(pmlb.regression_dataset_names):
        data = pmlb.fetch_data(dataset_name, return_X_y=False, local_cache_dir=data_dir)
        datasets_dict[dataset_name] = {'features': data.shape[1], 'samples': data.shape[0]}
    df = pd.DataFrame(datasets_dict).T
    df.sort_values(by='features', ascending=False, inplace=True)
    df.to_csv(osp.join(out_path, 'datasets.csv'))
    logger.info('Finish download_regression_datasets in {:.3f} sec'.format(time.time() - t0))
