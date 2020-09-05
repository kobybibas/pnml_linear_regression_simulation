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

from learner_utils.analytical_pnml_utils import calc_analytical_pnml_performance
from learner_utils.learner_helpers import calc_var_with_valset
from learner_utils.mdl_utils import calc_mdl_performance
from learner_utils.minimum_norm_utils import calc_mn_learner_performance, calc_theta_mn
from learner_utils.pnml_utils import calc_empirical_pnml_performance, calc_genie_performance

logger = logging.getLogger(__name__)


def create_trainset_sizes_to_eval(trainset_sizes: list, n_train: int, n_features: int, num_trainset_sizes: int) -> list:
    """
    Create list of training set sizes to evaluate.
    :param n_train: Training set size.
    :param n_features:  Number of feature
    :param num_trainset_sizes: How much training set sizes to evaluate.
    :return: list of training set sizes.
    """
    if len(trainset_sizes) == 0:
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


def get_data(dataset_name: str, data_dir: str = None):
    x_all, y_all = pmlb.fetch_data(dataset_name, return_X_y=True, local_cache_dir=data_dir)
    return x_all, y_all


def split_dataset(x_all, y_all, is_standardize_feature: bool = False, is_standardize_samples: bool = True,
                  is_add_bias_term: bool = True):
    if is_standardize_samples is True:
        # Normalize the data. To match mdl-comp preprocess
        x_all, y_all = standardize_samples(x_all, y_all)

    # Split: train val test = [0.6, 0.2 ,0.2]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=True)

    if is_standardize_feature is True:
        # apply standardization on numerical features
        x_train, x_val, x_test = standardize_feature(x_train, x_val, x_test)

    if is_add_bias_term is True:
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
        x_val = np.hstack((x_val, np.ones((x_val.shape[0], 1))))
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    return x_train, y_train, x_val, y_val, x_test, y_test


@ray.remote
def execute_trail(x_all: np.ndarray, y_all: np.ndarray, trail_num: int, trainset_size: int, dataset_name: str,
                  is_eval_mdl: bool, is_eval_empirical_pnml: bool, is_eval_analytical_pnml: bool,
                  is_standardize_feature: bool, is_standardize_samples: bool, is_add_bias_term: bool,
                  pnml_params_dict: dict,
                  debug_print: bool = True, fast_dev_run: bool = False):
    t0 = time.time()

    # Execute trails
    df_list = []

    # Split dataset
    t1 = time.time()
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_all, y_all,
                                                                   is_standardize_feature=is_standardize_feature,
                                                                   is_standardize_samples=is_standardize_samples,
                                                                   is_add_bias_term=is_add_bias_term)
    x_train, y_train = x_train[:trainset_size, :], y_train[:trainset_size]
    if fast_dev_run is True:
        x_train, y_train = x_train[:2, :], y_train[:2]
        x_val, y_val = x_val[:2, :], y_val[:2]
        x_test, y_test = x_test[:3, :], y_test[:3]

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
    var = calc_var_with_valset(x_val, y_val, theta_mn)

    # Minimum norm learner
    t1 = time.time()
    mn_df = calc_mn_learner_performance(x_train, y_train, x_val, y_val, x_test, y_test)
    df_list.append(mn_df)
    debug_print and logger.info('calc_mn_learner_performance in {:.3f} sec'.format(time.time() - t1))

    # Genie
    t1 = time.time()
    genie_df, theta_test_genies, theta_val_genies, genie_var_dict = calc_genie_performance(x_train, y_train,
                                                                                           x_val, y_val,
                                                                                           x_test, y_test,
                                                                                           theta_mn)
    df_list.append(genie_df)
    debug_print and logger.info('calc_genie_performance in {:.3f} sec. x_train.shape={}'.format(time.time() - t1,
                                                                                                x_train.shape))

    # Empirical pNML learner
    t1 = time.time()
    if is_eval_empirical_pnml is True:
        pnml_df = calc_empirical_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test,
                                                  theta_val_genies, theta_test_genies, genie_var_dict)
        df_list.append(pnml_df)
    debug_print and logger.info('calc_empirical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # Analytical pNML learner
    t1 = time.time()
    if is_eval_analytical_pnml is True:
        var_genies = [var] * len(x_test)
        analytical_pnml_df = calc_analytical_pnml_performance(x_train, y_train, x_test, y_test,
                                                              theta_mn, theta_test_genies, var_genies)
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
