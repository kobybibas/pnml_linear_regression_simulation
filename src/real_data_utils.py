import logging
import time
from collections import defaultdict

import numpy as np
import numpy.linalg as npl
import pandas as pd
import pmlb
import ray
import scipy.optimize
import scipy.stats as st
import sklearn.model_selection
import sklearn.utils
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pnml_min_norm_utils import AnalyticalMinNormPNML

logger = logging.getLogger(__name__)


def standardize_samples(x_data: np.ndarray, y: np.ndarray):
    # Normalize the data. To match mdl-comp preprocess
    x_mean, x_std = np.mean(x_data, axis=1).reshape(-1, 1), np.std(x_data, axis=1).reshape(-1, 1)
    x_std[x_std < 1e-12] = 1
    x_data -= x_mean
    x_data /= x_std
    y = (y - np.mean(y)) / np.std(y)
    return x_data, y


def standardize_feature(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> (
        np.ndarray, np.ndarray, np.ndarray):
    x_train_stand = x_train.copy()
    x_val_stand = x_val.copy()
    x_test_stand = x_test.copy()

    for k in range(x_train.shape[1]):
        # fit on training data columns
        scale = StandardScaler().fit(x_train_stand[:, k].reshape(-1, 1))

        # transform the data column features (columns)
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
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.2,
                                                                                shuffle=True)
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.25,
                                                                              shuffle=True)

    if is_standardize_feature is True:
        # apply standardization on numerical features
        x_train, x_val, x_test = standardize_feature(x_train, x_val, x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def compute_practical_mdt_comp(x_train, y_train, variance: float = 1.0):
    """
    Calculate prac-mdl-comp for this dataset
    :param x_train: 
    :param y_train: 
    :param variance: 
    :return: 
    """

    # My addition: npl.eig -> npl.eigh
    eigenvals, eigenvecs = npl.eigh(x_train.T @ x_train)

    def calc_thetahat(l):
        inv = npl.pinv(x_train.T @ x_train + l * np.eye(x_train.shape[1]))
        return inv @ x_train.T @ y_train

    def prac_mdl_comp_objective(l):
        thetahat = calc_thetahat(l)
        mse_norm = npl.norm(y_train - x_train @ thetahat) ** 2 / (2 * variance)
        theta_norm = npl.norm(thetahat) ** 2 / (2 * variance)
        eigensum = 0.5 * np.sum(np.log((eigenvals + l) / l))
        return (mse_norm + theta_norm + eigensum) / y_train.size

    opt_solved = scipy.optimize.minimize(prac_mdl_comp_objective, x0=1e-10)
    prac_mdl = opt_solved.fun
    lambda_opt = opt_solved.x
    thetahat = calc_thetahat(lambda_opt)

    return {'prac_mdl': prac_mdl, 'lambda_opt': lambda_opt, 'thetahat': thetahat}


def calc_theta_mn(x_arr: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    # x_arr: each row is feature vec
    inv = np.linalg.pinv(x_arr.T @ x_arr)
    theta = inv @ x_arr.T @ y_vec
    return theta


def compute_mse(x_data: np.ndarray, y_labels: np.ndarray, theta: np.ndarray) -> float:
    y_hat_labels = x_data @ theta
    return float(np.mean((y_hat_labels - y_labels) ** 2))


def compute_logloss(x_arr: np.ndarray, y: np.ndarray, theta: np.ndarray, var: float) -> float:
    y_hat = x_arr @ theta
    prob = np.exp(-(y_hat - y) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss.mean()


def compute_pnml_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta: np.ndarray, var: float,
                         nfs: np.ndarray) -> float:
    y_hat = np.diag(x_arr @ theta)  # the diagonal corresponds to the genie predictions
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    prob /= nfs  # normalize by the pnml normalization factor
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss.mean()


def compute_sigma_with_valset(x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray) -> float:
    y_hat = x_val @ theta
    noise_std = np.std(y_hat - y_val)
    return noise_std


def estimate_sigma_unbiased(x_train: np.ndarray, y_train: np.ndarray, theta: np.ndarray, n_train: int,
                            num_features: int) -> float:
    y_hat = x_train @ theta
    noise_std = np.sum(np.square(y_train - y_hat)) / (n_train - num_features - 1)
    return noise_std


@ray.remote
def execute_trails(x_all: np.ndarray, y_all: np.ndarray, n_trails: int, trainset_size: int,
                   dataset_name: str, task_index: int, num_tasks: int,
                   is_eval_mdl: bool, is_adaptive_var: bool,
                   is_standardize_feature: bool, is_standardize_samples: bool):
    t0 = time.time()
    res_dict_mn = defaultdict(list)
    res_dict_pnml = defaultdict(list)
    res_dict_pnml_isit = defaultdict(list)
    res_dict_mdl = defaultdict(list)

    for i in range(n_trails):
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_all, y_all,
                                                                       is_standardize_feature=is_standardize_feature,
                                                                       is_standardize_samples=is_standardize_samples)
        x_train, y_train = x_train[:trainset_size, :], y_train[:trainset_size]
        theta_mn = calc_theta_mn(x_train, y_train)

        n_train, num_features = x_train.shape
        if n_train > num_features + 1:
            noise_std = estimate_sigma_unbiased(x_train, y_train, theta_mn, n_train, num_features)
        else:
            if is_adaptive_var is True:
                noise_std = compute_sigma_with_valset(x_val, y_val, theta_mn)  # todo: maybe set to default?
            else:
                noise_std = 1
        var = noise_std ** 2

        # Minimum norm
        res_dict_mn['test_mse'].append(compute_mse(x_test, y_test, theta_mn))
        res_dict_mn['train_mse'].append(compute_mse(x_train, y_train, theta_mn))
        res_dict_mn['theta_norm'].append(np.mean(theta_mn ** 2))
        res_dict_mn['test_logloss'].append(compute_logloss(x_test, y_test, theta_mn, var))
        res_dict_mn['variance'].append(compute_sigma_with_valset(x_val, y_val, theta_mn))

        # pNML

        # Fit genie
        pnml_h = AnalyticalMinNormPNML(x_train, theta_mn)
        if num_features >= trainset_size:
            # over param region
            theta_genies = [pnml_h.fit_overparam_genie(theta_mn, x_train, y_train, x.T, y) for x, y in
                            zip(x_test, y_test)]
        else:
            # under param region
            theta_genies = [pnml_h.fit_underparam_genie(x_train, y_train, x.T, y) for x, y in zip(x_test, y_test)]
        theta_genies = np.array(theta_genies).T  # each column is a different genie

        norm_factors = np.array([pnml_h.calc_over_param_norm_factor(x.T, var) for x in x_test])
        res_dict_pnml['regret'].append(np.log(norm_factors).mean())
        res_dict_pnml['test_logloss'].append(compute_pnml_logloss(x_test, y_test, theta_genies, var, norm_factors))

        # pNML isit
        norm_factors = np.array([pnml_h.calc_under_param_norm_factor(x.T) for x in x_test])
        res_dict_pnml_isit['regret'].append(np.log(norm_factors).mean())
        res_dict_pnml_isit['test_logloss'].append(compute_pnml_logloss(x_test, y_test, theta_genies, var, norm_factors))

        # MDL
        if is_eval_mdl is True:
            mdl_dict = compute_practical_mdt_comp(x_train, y_train, variance=var)
            theta_mdl = mdl_dict['thetahat']
            lambda_opt = mdl_dict['lambda_opt']

            res_dict_mdl['lambda_opt'].append(lambda_opt)
            res_dict_mdl['test_mse'].append(compute_mse(x_test, y_test, theta_mdl))
            res_dict_mdl['train_mse'].append(compute_mse(x_train, y_train, theta_mdl))
            res_dict_mdl['theta_norm'].append(np.mean(theta_mdl ** 2))
            res_dict_mdl['test_logloss'].append(compute_logloss(x_test, y_test, theta_mdl, var))
            res_dict_mdl['variance'].append(compute_sigma_with_valset(x_val, y_val, theta_mdl))

    # General statistics
    res_dict = {'dataset_name': dataset_name,
                'trainset_size': x_train.shape[0],
                'valset_size': x_val.shape[0],
                'testset_size': x_test.shape[0],
                'num_features': x_train.shape[1],
                'n_trails': n_trails,
                'time': time.time() - t0,
                'task_index': task_index,
                'num_tasks': num_tasks
                }

    for key, values in res_dict_mn.items():
        res_dict[key + '_mn_mean'] = np.mean(values)
        res_dict[key + '_mn_std'] = np.std(values)
        res_dict[key + '_mn_sem'] = st.sem(values)

    for key, values in res_dict_pnml.items():
        res_dict[key + '_pnml_mean'] = np.mean(values)
        res_dict[key + '_pnml_std'] = np.std(values)
        res_dict[key + '_pnml_sem'] = st.sem(values)

    for key, values in res_dict_pnml_isit.items():
        res_dict[key + '_pnml_isit_mean'] = np.mean(values)
        res_dict[key + '_pnml_isit_std'] = np.std(values)
        res_dict[key + '_pnml_isit_sem'] = st.sem(values)

    if is_eval_mdl:
        for key, values in res_dict_mdl.items():
            res_dict[key + '_mdl_mean'] = np.mean(values)
            res_dict[key + '_mdl_std'] = np.std(values)
            res_dict[key + '_mdl_sem'] = st.sem(values)

    return res_dict


def download_regression_datasets(data_dir: str) -> pd.DataFrame:
    datasets_dict = {}
    for dataset_name in tqdm(pmlb.regression_dataset_names):
        data = pmlb.fetch_data(dataset_name, return_X_y=False, local_cache_dir=data_dir)
        datasets_dict[dataset_name] = {'features': data.shape[1], 'samples': data.shape[0]}
    df = pd.DataFrame(datasets_dict).T
    df.sort_values(by='features', ascending=False, inplace=True)
    return df
