import logging
import time

import numpy as np
import pandas as pd
import scipy.optimize as optimize

from learner_utils.learner_helpers import calc_best_var, calc_var_with_valset
from learner_utils.learner_helpers import calc_logloss, calc_mse, calc_theta_norm, fit_least_squares_estimator
from learner_utils.optimization_utils import fit_norm_constrained_least_squares

logger = logging.getLogger(__name__)


def add_test_to_train(phi_train: np.ndarray, phi_test: np.ndarray) -> np.ndarray:
    """
    Add the test set feature to training set feature matrix
    :param phi_train: training set feature matrix.
    :param phi_test: test set feature.
    :return: concat train and test.
    """
    # Make the input as row vector
    if len(phi_test.shape) == 1:
        phi_test = np.expand_dims(phi_test, 0)
    phi_arr = np.concatenate((phi_train, phi_test), axis=0)
    return phi_arr


def compute_pnml_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta_genies: np.ndarray, genie_vars: list,
                         nfs: np.ndarray) -> float:
    genie_vars = np.array(genie_vars)
    y_hat = np.array([x @ theta_genie for x, theta_genie in zip(x_arr, theta_genies)]).squeeze()
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * genie_vars)) / np.sqrt(2 * np.pi * genie_vars)

    # Normalize by the pnml normalization factor
    prob /= nfs
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss


def optimize_pnml_var(epsilon_square_true, epsilon_square_list, y_trained_list) -> np.ndarray:
    # """
    # Fit least squares estimator. Constrain it by the max norm constrain
    # :param phi_arr: data matrix. Each row represents an example.
    # :param y: label vector
    # :param max_norm: the constraint of the fitted parameters.
    # :return: fitted parameters that satisfy the max norm constrain
    # """

    # y_hat = np.array([theta_i.T @ phi_test for theta_i in theta_genie]).squeeze()
    # y_trained.squeeze()
    # epsilon_square = -(y_test - y_hat) ** 2
    # epsilon_true_square =(y_test - theta_genie.T @ phi_test)**2
    def calc_nf(sigma_fit):
        var_fit = sigma_fit ** 2
        # Genie probs
        genies_probs = np.exp(-epsilon_square_list / (2 * var_fit)) / np.sqrt(2 * np.pi * var_fit)

        # Normalization factor
        nf = 2 * np.trapz(genies_probs, x=y_trained_list)
        return nf

    def calc_jac(sigma_fit):
        var_fit = sigma_fit ** 2
        nf = calc_nf(sigma_fit)

        jac = (1 / (2 * nf * var_fit ** 2)) * (var_fit - nf * epsilon_square_true)
        return jac

    def calc_loss(sigma_fit):
        var_fit = sigma_fit ** 2

        # Genie probs
        nf = calc_nf(sigma_fit)

        loss = 0.5 * np.log(2 * np.pi * var_fit) + epsilon_square_true / (2 * var_fit) + np.log(nf)
        return loss

    # Optimize
    sigma_0 = 0.001
    res = optimize.minimize(calc_loss, sigma_0, jac=calc_jac)

    # Verify output
    sigma = res.x
    var = float(sigma ** 2)

    if bool(res.success) is False and \
            not res.message == 'Desired error not necessarily achieved due to precision loss.':
        logger.warning(res.message)
    return var


class Pnml:
    def __init__(self, phi_train: np.ndarray, y_train: np.ndarray, lamb: float = 0.0, var: float = 1e-6):
        # Minimum variance for prediction
        self.var = var

        # dict of the genies output:
        #     y_vec: labels that were evaluated, probs:probability of the corresponding labels
        self.genies_output = {'y_vec': None, 'probs': None, 'epsilons': None, 'theta_genie': None}
        self.norm_factor = None

        # The interval for possible y, for creating pdf
        self.y_to_eval = np.arange(-1000, 1000, 0.01)

        # Train feature matrix and labels
        self.phi_train = phi_train
        self.y_train = y_train

        # Regularization term
        self.lamb = lamb

        # ERM least squares parameters
        self.theta_erm = fit_least_squares_estimator(self.phi_train, self.y_train, lamb=self.lamb)

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray):
        return fit_least_squares_estimator(phi_arr, y, lamb=self.lamb)

    def set_y_interval(self, dy_min: float, y_max: float, y_num: int, is_adaptive: bool = False, logbase=10):
        """
        Build list on which the probability will be evaluated.
        :param dy_min: first evaluation point after 0.
        :param y_max: the higher bound of the interval.
        :param y_num: number of points to evaluate
        :param is_adaptive: gives more dy around zero.
        :return: y list on which to eval the probability.
        """
        # Initialize evaluation interval
        assert dy_min > 0

        y_to_eval = np.append([0], np.logspace(np.log(dy_min) / np.log(logbase),
                                               np.log(y_max) / np.log(logbase), int(y_num / 2), base=logbase))
        y_to_eval = np.unique(np.concatenate((-y_to_eval, y_to_eval)))

        if is_adaptive is True:
            y_to_eval = np.concatenate((np.arange(0, 0.001, 1e-7),
                                        np.arange(0.001, 1, 1e-3),
                                        np.arange(1.0, 10, 0.1),
                                        np.arange(10, 100, 1.0),
                                        np.arange(100, 1000, 10.0)))
            y_to_eval = np.unique(np.concatenate((-y_to_eval, y_to_eval)))
        self.y_to_eval = y_to_eval

    def predict_erm(self, phi_test: np.ndarray) -> float:
        if self.theta_erm is None:
            self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.y_train)
        return float(self.theta_erm.T @ phi_test)

    def calc_norm_factor(self, phi_test: np.array, variance: float = None) -> float:
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features to evaluate.
        :param variance: genie's variance.
        :return: log normalization factor.
        """
        self.reset_test_sample()
        if variance is None:
            variance = self.var

        y_vec = self.create_y_vec_to_eval(phi_test)
        thetas = self.calc_genie_thetas(phi_test, y_vec)

        # Calc genies predictions
        genies_probs = self.calc_genies_probs(phi_test, y_vec, thetas, variance)

        # Integrate to find the pNML normalization factor
        norm_factor = np.trapz(genies_probs, x=y_vec)
        return norm_factor

    def create_y_vec_to_eval(self, phi_test):
        # Predict around the ERM prediction
        y_pred = self.theta_erm.T @ phi_test
        y_vec = self.y_to_eval + y_pred
        self.genies_output['y_vec'] = y_vec
        return y_vec

    def calc_genie_thetas(self, phi_test, y_vec):
        phi_arr = add_test_to_train(self.phi_train, phi_test)
        thetas = [self.fit_least_squares_estimator(phi_arr, np.append(self.y_train, y)) for y in y_vec]
        return thetas

    def calc_genies_probs(self, phi_test, y_trained: np.ndarray, thetas: np.ndarray, variance: float) -> np.ndarray:
        """
        Calculate the genie probability of the label it was trained with
        :param y_trained: The labels that the genie was trained with
        :param thetas: The fitted parameters to the label (the trained genie)
        :param variance: the variance (sigma^2)
        :return: the genie probability of the label it was trained with
        """
        y_hat = np.array([theta.T @ phi_test for theta in thetas]).squeeze()
        y_trained = y_trained.squeeze()
        genies_probs = np.exp(-(y_trained - y_hat) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        self.genies_output['y_hat'] = y_hat
        self.genies_output['probs'] = genies_probs
        return genies_probs

    def reset_test_sample(self):
        self.genies_output = {}


class PnmlMinNorm(Pnml):
    def __init__(self, constrain_factor, *args, **kargs):
        # Initialize all other class var
        super().__init__(*args, **kargs)

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

        # Fitted least squares parameters with norm constraint
        self.max_norm = self.constrain_factor * calc_theta_norm(self.theta_erm)

    def set_constrain_factor(self, constrain_factor: float):
        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray) -> np.ndarray:
        max_norm = self.max_norm
        theta = fit_norm_constrained_least_squares(phi_arr, y, max_norm)
        return theta


def verify_empirical_pnml_results(pnml_h, norm_factor: float, x_train, test_idx: int) -> (bool, str):
    is_failed = False
    message = ''

    # Some check for converges:
    if norm_factor < 1.0:
        # Expected positive regret
        message += 'Negative regret={:.3f}. [x_train.shape idx]=[{} {}]'.format(np.log(norm_factor), x_train.shape,
                                                                                test_idx)
        is_failed = True
    if pnml_h.genies_output['probs'][-1] > np.finfo('float').eps:
        # Expected probability 0 at the edges
        message += 'Interval is too small prob={}. [x_train.shape idx]=[{} {}]'.format(
            pnml_h.genies_output['probs'][-1], x_train.shape, test_idx)
        is_failed = True
    return is_failed, message


def execute_pnml_on_sample(pnml_h: Pnml, x_i, y_i, theta_genie_i, var_dict, i: int, total: int):
    t0 = time.time()

    # Calc normalization factor
    pnml_h.reset_test_sample()
    y_vec = pnml_h.create_y_vec_to_eval(x_i)
    thetas = pnml_h.calc_genie_thetas(x_i, y_vec)

    # Calc best sigma
    epsilon_square_list = (y_vec - np.array([theta.T @ x_i for theta in thetas]).squeeze()) ** 2
    epsilon_square_true = (y_i - theta_genie_i.T @ x_i) ** 2
    var_best = optimize_pnml_var(epsilon_square_true, epsilon_square_list, y_vec)
    var_dict['pnml_best_var'] = var_best

    # Calc genies predictions
    nf_dict = {}
    for var_name, var in var_dict.items():
        genies_probs = pnml_h.calc_genies_probs(x_i, y_vec, thetas, var)
        nf = 2 * np.trapz(genies_probs, x=y_vec)

        is_failed, message = verify_empirical_pnml_results(pnml_h, nf, pnml_h.phi_train, i)
        msg  = '[{:03d}/{}] \t nf={:.8f}. \t {:.1f} sec. x_train.shape={} \t is_failed={}. {}={} {}'.format(
            i, total - 1, nf, time.time() - t0, pnml_h.phi_train.shape, is_failed, var_name, var, message)
        if is_failed is True:
            logger.warning(msg)
        else:
            logger.info(msg)
        nf_dict[var_name] = nf

    return nf_dict


def calc_empirical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray,
                                    x_val: np.ndarray, y_val: np.ndarray,
                                    x_test: np.ndarray, y_test: np.ndarray,
                                    theta_val_genies: np.ndarray, theta_test_genies: np.ndarray,
                                    genie_var_dict: dict) -> pd.DataFrame:
    n_train, n_features = x_train.shape

    # Initialize pNML
    n_train_effective = n_train + 1  # We add the test sample data to training
    if n_train_effective > n_features:
        pnml_h = Pnml(phi_train=x_train, y_train=y_train, lamb=0.0, var=1e-3)
    else:
        pnml_h = PnmlMinNorm(constrain_factor=1.0, phi_train=x_train, y_train=y_train, lamb=0.0, var=1e-3)

    # Set interval
    y_to_eval = np.append(0, np.logspace(-16, 4, 1000))  # One side interval
    pnml_h.y_to_eval = y_to_eval

    # Compute best variance using validation set
    best_vars = []
    for i, (x_i, y_i, theta_genie_i) in enumerate(zip(x_val, y_val, theta_val_genies)):
        nf_dict_i = execute_pnml_on_sample(pnml_h, x_i, y_i, theta_genie_i, {}, i, len(y_test))
        best_var = nf_dict_i['pnml_best_var']
        best_vars.append(best_var)
    genie_var_dict['empirical_pnml_valset_mean_var'] = [float(np.mean(best_vars))] * len(x_test)
    genie_var_dict['empirical_pnml_valset_median_var'] = [float(np.median(best_vars))] * len(x_test)

    # Execute on test set
    nf_dict = {var_type: [] for var_type in genie_var_dict.keys()}
    for i, (x_i, y_i, theta_genie_i) in enumerate(zip(x_test, y_test, theta_test_genies)):
        var_dict = {var_name: var_values[i] for var_name, var_values in genie_var_dict.items()}
        nf_dict_i = execute_pnml_on_sample(pnml_h, x_i, y_i, theta_genie_i, var_dict, i, len(y_test))
        for nf_name, nf_value in nf_dict_i.items():
            if nf_name not in nf_dict:
                nf_dict[nf_name] = []
            nf_dict[nf_name].append(nf_value)

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


def fit_genies_to_dataset(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                          theta_erm: np.ndarray) -> (list, list):
    n_train, num_features = x_train.shape

    theta_genies, var_genies = [], []
    for x_test_i, y_test_i in zip(x_test, y_test):

        # Add test to train
        phi_arr, y = add_test_to_train(x_train, x_test_i), np.append(y_train, y_test_i)
        assert phi_arr.shape[0] == len(y)

        n_train_effective = n_train + 1  # We add the test sample data to training todo: check if fixed
        if n_train_effective > num_features:
            # Fit under param region
            theta_genie_i = fit_least_squares_estimator(phi_arr, y, lamb=0.0)
        else:
            # Fit over param region
            constrain = calc_theta_norm(theta_erm)
            theta_genie_i = fit_norm_constrained_least_squares(phi_arr, y, constrain)

        # Optimize genie variance
        genie_var_i = calc_best_var(x_test_i, y_test_i, theta_genie_i)

        theta_genies.append(theta_genie_i)
        var_genies.append(genie_var_i)
    return theta_genies, var_genies


def calc_genie_performance(x_train: np.ndarray, y_train: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray,
                           theta_erm: np.ndarray) -> (pd.DataFrame, list, dict):
    theta_mn = fit_least_squares_estimator(x_train, y_train, lamb=0.0)

    mn_var = calc_var_with_valset(x_val, y_val, theta_mn)
    theta_test_genies, adaptive_var = fit_genies_to_dataset(x_train, y_train, x_test, y_test, theta_erm)
    theta_val_genies, valset_var = fit_genies_to_dataset(x_train, y_train, x_val, y_val, theta_erm)

    # Different vars to experiment
    n_test = len(x_test)
    var_dict = {'valset_mean_var': [np.mean(valset_var)] * n_test,
                'adaptive_var_var': adaptive_var,
                'mn_var_var': [mn_var] * n_test,
                'valset_median_var': [np.median(valset_var)] * n_test}

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
