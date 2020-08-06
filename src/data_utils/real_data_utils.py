import logging
import os.path as osp
import time

import numpy as np
import pandas as pd
import pmlb
import ray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from learner_utils.analytical_pnml_utils import calc_analytical_pnml_performance, calc_genie_performance
from learner_utils.learner_helpers import estimate_variance_with_valset
from learner_utils.mdl_utils import calc_mdl_performance
from learner_utils.minimum_norm_utils import calc_mn_learner_performance, calc_theta_mn
from learner_utils.pnml_utils import calc_empirical_pnml_performance

logger = logging.getLogger(__name__)


def create_trainset_sizes_to_eval(n_train: int, n_features: int, num_trainset_sizes: int) -> list:
    """
    Create list of training set sizes to evaluate.
    :param n_train: Training set size.
    :param n_features:  Number of feature
    :param num_trainset_sizes: How much training set sizes to evaluate.
    :return: list of training set sizes.
    """
    trainset_sizes = np.logspace(np.log10(2), np.log10(n_train), num_trainset_sizes).astype(int)
    trainset_sizes = np.append(trainset_sizes, n_features)
    trainset_sizes = np.unique(trainset_sizes)
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


@ray.remote
def execute_trail(x_all: np.ndarray, y_all: np.ndarray, trail_num: int, trainset_size: int, dataset_name: str,
                  is_eval_mdl: bool, is_eval_empirical_pnml: bool, is_eval_analytical_pnml: bool,
                  is_standardize_feature: bool, is_standardize_samples: bool, debug_print: bool = True):
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
    var = estimate_variance_with_valset(x_val, y_val, theta_mn)

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
        pnml_df = calc_empirical_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test, theta_mn)
        df_list.append(pnml_df)
    debug_print and logger.info('calc_empirical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # Analytical pNML learner
    t1 = time.time()
    if is_eval_analytical_pnml is True:
        analytical_pnml_df = calc_analytical_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test,
                                                              theta_genies, theta_mn)
        df_list.append(analytical_pnml_df)
    debug_print and logger.info('calc_analytical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

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
