import logging
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pmlb
import ray
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from learner_classes.analytical_pnml_utils import calc_analytical_pnml_performance
from learner_classes.learner_utils import estimate_base_var
from learner_classes.mdl_utils import calc_mdl_performance
from learner_classes.minimum_norm_utils import calc_mn_learner_performance, calc_theta_mn
from learner_classes.pnml_utils import calc_empirical_pnml_performance

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
    trainset_sizes = np.unique(np.linspace(5, n_features, num_trainset_sizes_over_param).astype(int))
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


@ray.remote
def execute_trails(x_all: np.ndarray, y_all: np.ndarray, n_trails: int, trainset_size: int,
                   dataset_name: str, task_index: int, num_tasks: int,
                   is_eval_mdl: bool, is_eval_empirical_pnml: bool, is_eval_analytical_pnml: bool,
                   is_adaptive_var: bool,
                   is_standardize_feature: bool, is_standardize_samples: bool,
                   debug_print: bool = True):
    t0 = time.time()

    # Execute trails
    res_trails_dict = {}
    for i in range(n_trails):

        # Split dataset
        t1 = time.time()
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_all, y_all,
                                                                       is_standardize_feature=is_standardize_feature,
                                                                       is_standardize_samples=is_standardize_samples)
        x_train, y_train = x_train[:trainset_size, :], y_train[:trainset_size]
        debug_print and logger.info('split_dataset in {:.3f} sec'.format(time.time() - t1))

        # Compute variance
        theta_mn = calc_theta_mn(x_train, y_train)
        var = estimate_base_var(x_train, y_train, x_val, y_val, theta_mn, is_adaptive_var=is_adaptive_var)

        # Minimum norm learner
        t1 = time.time()
        res_dict_mn = calc_mn_learner_performance(x_train, y_train, x_val, y_val, x_test, y_test, theta_mn=theta_mn)
        add_trail_to_res_dict(res_trails_dict, res_dict_mn, 'mn')
        debug_print and logger.info('calc_mn_learner_performance in {:.3f} sec'.format(time.time() - t1))

        # Empirical pNML learner
        t1 = time.time()
        theta_genies = []  # Fill this by the calc_empirical_pnml_performance func
        if is_eval_empirical_pnml is True:
            res_pnml = calc_empirical_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test,
                                                       theta_mn, theta_genies)
            add_trail_to_res_dict(res_trails_dict, res_pnml, 'empirical_pnml')
            theta_genies = theta_genies[0]
        debug_print and logger.info('calc_empirical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

        # Analytical pNML learner
        t1 = time.time()
        if is_eval_analytical_pnml is True:
            res_pnml, res_pnml_isit, res_genie = calc_analytical_pnml_performance(x_train, y_train, x_val, y_val,
                                                                                  x_test, y_test, theta_mn,
                                                                                  theta_genies if len(
                                                                                      theta_genies) > 0 else None)
            add_trail_to_res_dict(res_trails_dict, res_pnml, 'analytical_pnml')
            add_trail_to_res_dict(res_trails_dict, res_pnml, 'analytical_pnml_isit')
            add_trail_to_res_dict(res_trails_dict, res_pnml, 'genie')
        debug_print and logger.info('calc_analytical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

        # MDL
        t1 = time.time()
        if is_eval_mdl is True:
            res_dict_mdl = calc_mdl_performance(x_train, y_train, x_val, y_val, x_test, y_test, var)
            add_trail_to_res_dict(res_trails_dict, res_dict_mdl, 'mdl')
        debug_print and logger.info('calc_mdl_performance in {:.3f} sec'.format(time.time() - t1))

    # General statistics
    res_dict = {'dataset_name': dataset_name,
                'trainset_size': x_train.shape[0],
                'valset_size': x_val.shape[0],
                'testset_size': x_test.shape[0],
                'num_features': x_train.shape[1],
                'n_trails': n_trails,
                'time': time.time() - t0,
                'task_index': task_index,
                'num_tasks': num_tasks}

    # Summarize results
    t1 = time.time()
    for method_name, method_dict in res_trails_dict.items():
        for key, values in method_dict.items():
            res_dict[key + f'_{method_name}_mean'] = np.mean(values)
            res_dict[key + f'_{method_name}_std'] = np.std(values)
            res_dict[key + f'_{method_name}_sem'] = st.sem(values)
    debug_print and logger.info('Summarize results in {:.3f} sec'.format(time.time() - t1))

    debug_print and logger.info(
        '    Finish {} trainset_size: {} in {:.3f}'.format(dataset_name, trainset_size, time.time() - t0))
    return res_dict


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
