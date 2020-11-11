import logging
import os.path as osp
import time
from glob import glob

import numpy as np
import numpy.linalg as npl
import pandas as pd
import ray
from sklearn.preprocessing import StandardScaler

from learner_utils.analytical_pnml_utils import calc_analytical_pnml_performance
from learner_utils.mdl_utils import calc_mdl_performance
from learner_utils.minimum_norm_utils import calc_mn_learner_performance
from learner_utils.pnml_real_data_helper import calc_empirical_pnml_performance, calc_genie_performance

logger = logging.getLogger(__name__)


def create_trainset_sizes_to_eval(trainset_sizes: list, n_train: int, n_features: int) -> list:
    """
    Create list of training set sizes to evaluate if was not predefined in the cfg file.
    :param trainset_sizes: Predefined training set sizes.
    :param n_train: Training set size.
    :param n_features:  Number of feature
    :return: list of training set sizes.
    """
    if len(trainset_sizes) == 0:
        trainset_sizes_over_param = np.arange(4, n_features + 1).astype(int)
        trainset_sizes_under_param = np.logspace(np.log10(n_features + 1), np.log10(n_train), 10).astype(int)
        trainset_sizes = np.append(trainset_sizes_over_param, trainset_sizes_under_param)
        trainset_sizes = np.unique(trainset_sizes)
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


def get_available_train_test_splits(dataset_name: str, data_dir: str) -> np.ndarray:
    # Initialize output
    splits = []

    # Existing files
    paths = glob(osp.join(data_dir, dataset_name, 'data', f'index_train_*.txt'))

    # For each file name, extract its split num.
    for path in paths:
        split = int(path.split('_')[-1][:-4])  # Get split number
        splits.append(split)
    return np.sort(splits)


def get_uci_data(dataset_name: str, data_dir: str, train_test_split_num: int,
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
    index_train_path = osp.join(data_dir, dataset_name, 'data', f'index_train_{train_test_split_num}.txt')
    index_test_path = osp.join(data_dir, dataset_name, 'data', f'index_test_{train_test_split_num}.txt')
    index_train = np.loadtxt(index_train_path).astype(int)
    index_test = np.loadtxt(index_test_path).astype(int)

    # Train-test split
    x_train, y_train = x_all[index_train], y_all[index_train]
    x_test, y_test = x_all[index_test], y_all[index_test]

    # Train-val split
    num_training_examples = int(0.9 * x_train.shape[0])
    x_val, y_val = x_train[num_training_examples:, :], y_train[num_training_examples:]
    x_train, y_train = x_train[:num_training_examples, :], y_train[:num_training_examples]

    x_val, y_val = x_val[:100, :], y_val[:100]  # Maximum of 100 validation samples
    if is_standardize_features is True:
        # Apply standardization on numerical features
        x_train, x_val, x_test = standardize_features(x_train, x_val, x_test)
        y_train, y_val, y_test = [y_set.squeeze() for y_set in
                                  standardize_features(y_train.reshape(-1, 1), y_val.reshape(-1, 1),
                                                       y_test.reshape(-1, 1))]

    if is_add_bias_term is True:
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
        x_val = np.hstack((x_val, np.ones((x_val.shape[0], 1))))
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    if is_normalize_data is True:
        x_train = x_train / npl.norm(x_train, axis=1, keepdims=True)
        x_val = x_val / npl.norm(x_val, axis=1, keepdims=True)
        x_test = x_test / npl.norm(x_test, axis=1, keepdims=True)

    return x_train, y_train, x_val, y_val, x_test, y_test


@ray.remote
def execute_trail(x_train: np.ndarray, y_train: np.ndarray,
                  x_val: np.ndarray, y_val: np.ndarray,
                  x_test: np.ndarray, y_test: np.ndarray,
                  trail_num: int, trainset_size: int, dataset_name: str,
                  optimization_dict: dict, debug_print: bool = True, logger_file_path: str = None) -> pd.DataFrame:
    # Initialize output
    logger = logging.getLogger(__name__)
    if logger_file_path is not None:
        logging.basicConfig(filename=logger_file_path, level=logging.INFO)
        logger = logging.getLogger(__name__)

    t0 = time.time()
    df_list = []

    # General statistics
    n_test = len(x_test)
    param_df = pd.DataFrame({'dataset_name': [dataset_name] * n_test,
                             'trainset_size': [x_train.shape[0]] * n_test,
                             'trail_num': [trail_num] * n_test,
                             'valset_size': [x_val.shape[0]] * n_test,
                             'testset_size': [x_test.shape[0]] * n_test,
                             'num_features': [x_train.shape[1]] * n_test})
    df_list.append(param_df)

    # Minimum norm learner
    t1 = time.time()
    mn_df, theta_mn, mn_valset_var = calc_mn_learner_performance(x_train, y_train, x_val, y_val, x_test, y_test)
    df_list.append(mn_df)
    debug_print and logger.info('calc_mn_learner_performance in {:.3f} sec'.format(time.time() - t1))

    # Empirical pNML learner
    t1 = time.time()
    pnml_df = calc_empirical_pnml_performance(x_train, y_train, x_val, y_val, x_test, y_test,
                                              optimization_dict['pnml_lambda_optim_dict'])
    df_list.append(pnml_df)
    debug_print and logger.info('calc_empirical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # Analytical pNML learner
    t1 = time.time()
    pnml_vars = pnml_df['empirical_pnml_variance']
    analytical_pnml_df = calc_analytical_pnml_performance(x_train, y_train, x_test, y_test, theta_mn, pnml_vars)
    df_list.append(analytical_pnml_df)
    debug_print and logger.info('calc_analytical_pnml_performance in {:.3f} sec'.format(time.time() - t1))

    # Genie learner
    t1 = time.time()
    pnml_vars = pnml_df['empirical_pnml_variance']
    genie_df = calc_genie_performance(x_train, y_train, x_test, y_test, theta_mn, pnml_vars)
    df_list.append(genie_df)
    debug_print and logger.info('calc_genie_performance in {:.3f} sec. x_train.shape={}'.format(time.time() - t1,
                                                                                                x_train.shape))

    if False:
        # MDL
        t1 = time.time()
        mdl_df = calc_mdl_performance(x_train, y_train, x_val, y_val, x_test, y_test, mn_valset_var)
        df_list.append(mdl_df)
        debug_print and logger.info('calc_mdl_performance in {:.3f} sec'.format(time.time() - t1))

    res_df = pd.concat(df_list, axis=1, sort=False)
    res_df['test_idx'] = res_df.index
    debug_print and logger.info(
        '    Finish {} trainset_size: {} in {:.3f}'.format(dataset_name, trainset_size, time.time() - t0))
    return res_df
