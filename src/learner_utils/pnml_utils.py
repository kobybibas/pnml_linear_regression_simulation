import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_best_var, calc_var_with_valset
from learner_utils.learner_helpers import calc_logloss, calc_mse, calc_theta_norm, fit_least_squares_estimator
from learner_utils.optimization_utils import fit_least_squares_with_max_norm_constrain

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


def compute_pnml_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta_genies: np.ndarray, var_genies: list,
                         nfs: np.ndarray) -> float:
    var_genies = np.array(var_genies)
    y_hat = np.array([x @ theta_genie for x, theta_genie in zip(x_arr, theta_genies)]).squeeze()
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * var_genies)) / np.sqrt(2 * np.pi * var_genies)

    # Normalize by the pnml normalization factor
    prob /= nfs
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss


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
        self.theta_erm = None

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

    @staticmethod
    def calc_genies_probs(y_trained: np.ndarray, y_hat: np.ndarray, variance: float) -> np.ndarray:
        """
        Calculate the genie probability of the label it was trained with
        :param y_trained: The labels that the genie was trained with
        :param y_hat: The predicted label by the trained genie
        :param variance: the variance (sigma^2)
        :return: the genie probability of the label it was trained with
        """
        genies_probs = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(
            -(y_trained.squeeze() - y_hat.squeeze()) ** 2 / (2 * variance))
        return genies_probs

    def calc_norm_factor(self, phi_test: np.array, variance: float = None) -> float:
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features to evaluate.
        :param variance: genie's variance.
        :return: log normalization factor.
        """
        if self.theta_erm is None:
            self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.y_train)
        if variance is None:
            variance = self.var

        phi_arr = add_test_to_train(self.phi_train, phi_test)

        # Predict around the ERM prediction
        y_pred = self.theta_erm.T @ phi_test
        y_vec = self.y_to_eval + y_pred

        # Calc genies predictions
        thetas = [self.fit_least_squares_estimator(phi_arr, np.append(self.y_train, y)) for y in y_vec]
        y_hats = np.array([theta.T @ phi_test for theta in thetas]).squeeze()
        genies_probs = self.calc_genies_probs(y_vec, y_hats, variance)

        # Integrate to find the pNML normalization factor
        norm_factor = np.trapz(genies_probs, x=y_vec)

        # The genies predictors
        self.genies_output = {'y_vec': y_vec, 'probs': genies_probs, 'epsilons': y_vec - y_hats}
        return norm_factor


class PnmlMinNorm(Pnml):
    def __init__(self, constrain_factor, *args, **kargs):
        # Initialize all other class var
        super().__init__(*args, **kargs)

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

        # Fitted least squares parameters with norm constraint
        self.lamb = 0.0
        self.theta_erm = fit_least_squares_estimator(self.phi_train, self.y_train)
        self.max_norm = self.constrain_factor * calc_theta_norm(self.theta_erm)

    def set_constrain_factor(self, constrain_factor: float):
        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray) -> np.ndarray:
        max_norm = self.max_norm
        theta = fit_least_squares_with_max_norm_constrain(phi_arr, y, max_norm)
        return theta


def verify_empirical_pnml_results(pnml_h, norm_factor: float, x_train, j: int, y_to_eval,
                                  is_plot_failed: bool = False) -> (bool, str):
    is_failed = False
    message = ''

    # Some check for converges:
    if norm_factor < 1.0:
        # Expected positive regret
        message += 'Negative regret={:.3f}. [x_train.shape idx]=[{} {}]'.format(np.log(norm_factor), x_train.shape, j)
        is_failed = True
    if pnml_h.genies_output['probs'][-1] > np.finfo('float').eps:
        # Expected probability 0 at the edges
        message += 'Interval is too small prob={}. [x_train.shape idx]=[{} {}]'.format(
            pnml_h.genies_output['probs'][-1], x_train.shape, j)
        is_failed = True
    if is_plot_failed is True and is_failed is True:
        debug_dir = '../output/debug'
        os.makedirs(debug_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(y_to_eval, pnml_h.genies_output['probs'], '*')
        axs[0].grid(True)

        # plot positive part
        idxs = np.where(y_to_eval > 0)[0]
        axs[1].plot(y_to_eval[idxs], pnml_h.genies_output['probs'][idxs], '*')
        axs[1].set_xscale('log')
        axs[1].grid(True)
        plt.title('norm_factor={:.5f}'.format(norm_factor))
        fig.tight_layout()
        out_path = f'{debug_dir}/{time.strftime("%Y%m%d_%H%M%S")}_debug_regret_{x_train.shape}_{j}.jpg'
        fig.savefig(out_path)
        plt.close(fig)
    return is_failed, message


def calc_empirical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                                    theta_genies: np.ndarray, genie_vars: list) -> pd.DataFrame:
    n_train, n_features = x_train.shape

    # Initialize pNML
    var = np.mean(genie_vars)
    if n_train > n_features:
        pnml_h = Pnml(phi_train=x_train, y_train=y_train, lamb=0.0, var=var)
    else:
        pnml_h = PnmlMinNorm(constrain_factor=1.0, phi_train=x_train, y_train=y_train, lamb=0.0, var=var)

    # Set interval
    y_to_eval = np.append(0, np.logspace(-16, 3, 1000))  # One side interval
    # y_to_eval = np.unique(np.concatenate((-y_to_eval, y_to_eval)))
    pnml_h.y_to_eval = y_to_eval

    nfs = []
    for j, (x_test_i, var_i) in enumerate(zip(x_test, genie_vars)):
        t0 = time.time()

        # Calc normalization factor
        nf = 2 * pnml_h.calc_norm_factor(x_test_i, var_i)  # Calculate one sided norm factor
        nfs.append(nf)

        is_failed, message = verify_empirical_pnml_results(pnml_h, nf, x_train, j, y_to_eval, is_plot_failed=True)
        logger.warning('[{}/{}] \t nf={:.8f}. \t {:.1f} sec. x_train.shape={} \t is_failed={}. {}'.format(
            j, len(y_test), nf, time.time() - t0, x_train.shape, is_failed, message))

    regrets = np.log(nfs).tolist()
    res_dict = {'empirical_pnml_regret': regrets,
                'empirical_pnml_test_logloss': compute_pnml_logloss(x_test, y_test, theta_genies, genie_vars, nfs)}
    df = pd.DataFrame(res_dict)
    return df


def calc_testset_genies(x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray, theta_erm: np.ndarray) -> (list, list):
    n_train, num_features = x_train.shape

    theta_genies, var_genies = [], []
    for x_test_i, y_test_i in zip(x_test, y_test):

        # Add test to train
        phi_arr, y = add_test_to_train(x_train, x_test_i), np.append(y_train, y_test_i)
        assert phi_arr.shape[0] == len(y)

        if n_train > num_features:
            # Fit under param region
            theta_genie_i = fit_least_squares_estimator(phi_arr, y, lamb=0.0)
        else:
            # Fit over param region
            constrain = calc_theta_norm(theta_erm)
            theta_genie_i = fit_least_squares_with_max_norm_constrain(phi_arr, y, constrain)

        # Optimize genie variance
        genie_var_i = calc_best_var(phi_arr, y, theta_genie_i)

        theta_genies.append(theta_genie_i)
        var_genies.append(genie_var_i)
    return theta_genies, var_genies


def calc_genie_performance(x_train: np.ndarray, y_train: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray,
                           theta_erm: np.ndarray) -> (pd.DataFrame, list, list):
    theta_mn = fit_least_squares_estimator(x_train, y_train, lamb=0.0)
    var = calc_var_with_valset(x_val, y_val, theta_mn)
    theta_genies, var_genies = calc_testset_genies(x_train, y_train, x_test, y_test, theta_erm)

    # Metric
    test_logloss_adaptive_var = [float(calc_logloss(x, y, theta_i, var_i)) for x, y, theta_i, var_i in
                                 zip(x_test, y_test, theta_genies, var_genies)]
    test_logloss = [float(calc_logloss(x, y, theta_i, var)) for x, y, theta_i in zip(x_test, y_test, theta_genies)]

    test_mse = [float(calc_mse(x, y, theta_i)) for x, y, theta_i in zip(x_test, y_test, theta_genies)]
    res_dict = {'genie_adaptive_var_test_logloss': test_logloss_adaptive_var,
                'genie_test_logloss': test_logloss,
                'genie_test_mse': test_mse,
                'genie_theta_norm': [calc_theta_norm(theta_i) for theta_i in theta_genies],
                'genie_adaptive_var_variance': var_genies,
                'genie_variance': [var] * len(var_genies)}
    df = pd.DataFrame(res_dict)

    return df, theta_genies, var_genies
