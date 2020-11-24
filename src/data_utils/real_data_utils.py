import logging
import os.path as osp
import time
from glob import glob

import numpy as np
import numpy.linalg as npl
import pandas as pd
import ray
from sklearn.preprocessing import StandardScaler

from learner_utils.minimum_norm_utils import calc_mn_learner_performance
from learner_utils.pnml_real_data_helper import calc_pnml_performance

logger_default = logging.getLogger(__name__)


def create_trainset_sizes_to_eval(trainset_sizes: list, n_train: int, n_features: int) -> list:
    """
    Create list of training set sizes to evaluate if was not predefined in the cfg file.
    :param trainset_sizes: Predefined training set sizes.
    :param n_train: Training set size.
    :param n_features:  Number of feature
    :return: list of training set sizes.
    """
    if len(trainset_sizes) == 0:
        min_trainset = 2
        trainset_sizes_over_param = np.arange(min_trainset, n_features + 1).astype(int)
        trainset_sizes_under_param = np.logspace(np.log10(n_features + 1), np.log10(n_train), 10).round()
        trainset_sizes = np.append(trainset_sizes_over_param, trainset_sizes_under_param)
        trainset_sizes = np.unique(trainset_sizes).astype(int)
    return trainset_sizes


def standardize_features(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> (np.ndarray,
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


def get_set_splits(dataset_name: str, data_dir: str) -> np.ndarray:
    # Initialize output
    splits = []

    # Existing files
    paths = glob(osp.join(data_dir, dataset_name, 'data', f'index_train_*.txt'))

    # For each file name, extract its split num.
    for path in paths:
        split = int(path.split('_')[-1][:-4])  # Get split number
        splits.append(split)
    return np.sort(splits)


def choose_samples_for_debug(cfg, trainset: tuple, valset: tuple, testset: tuple) -> (tuple, tuple, tuple):
    x_train, y_train = trainset
    x_val, y_val = valset
    x_test, y_test = testset

    # reduce sets
    if cfg.fast_dev_run is True:
        x_train, y_train = x_train[:3, :], y_train[:3]
        x_val, y_val = x_val[:2, :], y_val[:2]
        x_test, y_test = x_test[:2, :], y_test[:2]

    # choose specific test samples
    if len(cfg.test_idxs) > 0:
        x_test, y_test = x_test[cfg.test_idxs, :], y_test[cfg.test_idxs]

    # choose specific val samples
    if len(cfg.val_idxs) > 0:
        x_val, y_val = x_val[cfg.val_idxs, :], y_val[cfg.val_idxs]

    # Reduce test set size: increasing execution speed
    if cfg.max_test_samples > 0:
        x_test, y_test = x_test[:cfg.max_test_samples, :], y_test[:cfg.max_test_samples]
    if cfg.max_val_samples > 0:
        x_val, y_val = x_val[:cfg.max_val_samples, :], y_val[:cfg.max_val_samples]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_uci_data(dataset_name: str, data_dir: str, split: int,
                 is_standardize_features: bool = True, is_add_bias_term: bool = True, is_normalize_data: bool = True):
    data_txt_path = osp.join(data_dir, dataset_name, 'data', 'data.txt')
    data = np.loadtxt(data_txt_path)

    index_features_path = osp.join(data_dir, dataset_name, 'data', 'index_features.txt')
    index_features = np.loadtxt(index_features_path)

    index_target_path = osp.join(data_dir, dataset_name, 'data', 'index_target.txt')
    index_target = np.loadtxt(index_target_path)

    x_all = data[:, [int(i) for i in index_features.tolist()]]
    y_all = data[:, int(index_target.tolist())]

    # Load split file
    index_train_path = osp.join(data_dir, dataset_name, 'data', f'index_train_{split}.txt')
    index_test_path = osp.join(data_dir, dataset_name, 'data', f'index_test_{split}.txt')
    index_train = np.loadtxt(index_train_path).astype(int)
    index_test = np.loadtxt(index_test_path).astype(int)

    # Train-test split
    x_train, y_train = x_all[index_train], y_all[index_train]
    x_test, y_test = x_all[index_test], y_all[index_test]

    # Train-val split
    num_training_examples = int(0.9 * x_train.shape[0])
    x_val, y_val = x_train[num_training_examples:, :], y_train[num_training_examples:]
    x_train, y_train = x_train[:num_training_examples, :], y_train[:num_training_examples]

    # Apply standardization on numerical features
    if is_standardize_features is True:
        x_train, x_val, x_test = standardize_features(x_train, x_val, x_test)
        y_train, y_val, y_test = [y_set.squeeze() for y_set in
                                  standardize_features(y_train.reshape(-1, 1), y_val.reshape(-1, 1),
                                                       y_test.reshape(-1, 1))]

    if is_normalize_data is True:
        x_train = x_train / npl.norm(x_train, axis=1, keepdims=True)
        x_val = x_val / npl.norm(x_val, axis=1, keepdims=True)
        x_test = x_test / npl.norm(x_test, axis=1, keepdims=True)

    if is_add_bias_term is True:
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
        x_val = np.hstack((x_val, np.ones((x_val.shape[0], 1))))
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def execute_reduce_dataset(x_arr: np.ndarray, y_vec: np.ndarray, set_size: int) -> (np.ndarray, np.ndarray):
    """
    :param x_arr: data array, each row is a sample
    :param y_vec: label vector
    :param set_size: the desired set size
    :return: reduced set
    """
    x_reduced, y_reduced = np.copy(x_arr[:set_size]), y_vec[:set_size]
    return x_reduced, y_reduced


@ray.remote
def execute_trail(x_train: np.ndarray, y_train: np.ndarray,
                  x_val: np.ndarray, y_val: np.ndarray,
                  x_test: np.ndarray, y_test: np.ndarray,
                  split: int, trainset_size: int, dataset_name: str,
                  pnml_optim_param: dict, debug_print: bool = True, logger_file_path: str = None) -> pd.DataFrame:
    t0 = time.time()
    logger = logging.getLogger(__name__)
    if logger_file_path is not None:
        logging.basicConfig(filename=logger_file_path, level=logging.INFO)
        logger = logging.getLogger(__name__)

    # Initialize output
    df_list = []

    # General statistics
    n_test = len(x_test)
    param_df = pd.DataFrame({'dataset_name': [dataset_name] * n_test,
                             'trainset_size': [x_train.shape[0]] * n_test,
                             'split': [split] * n_test,
                             'valset_size': [x_val.shape[0]] * n_test,
                             'testset_size': [x_test.shape[0]] * n_test,
                             'num_features': [x_train.shape[1]] * n_test})
    df_list.append(param_df)

    # Minimum norm learner
    t1 = time.time()
    mn_df = calc_mn_learner_performance(x_train, y_train, x_val, y_val, x_test, y_test, logger=logger)
    df_list.append(mn_df)
    debug_print and logger.info('calc_mn_learner_performance in {:.3f} sec'.format(time.time() - t1))

    # Empirical pNML learner
    t1 = time.time()
    pnml_df = calc_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test, split, pnml_optim_param, logger)
    df_list.append(pnml_df)
    debug_print and logger.info('calc_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    res_df = pd.concat(df_list, axis=1, sort=False)
    res_df['test_idx'] = res_df.index
    debug_print and logger.info(
        '    Finish {} trainset_size: {} in {:.3f}'.format(dataset_name, trainset_size, time.time() - t0))
    return res_df
