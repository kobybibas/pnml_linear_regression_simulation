import logging
import time

import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_best_var, calc_var_with_valset
from learner_utils.learner_helpers import calc_logloss, calc_mse, calc_theta_norm, fit_least_squares_estimator
from learner_utils.optimization_utils import fit_norm_constrained_least_squares
from learner_utils.optimization_utils import optimize_pnml_var

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

    # Concat train and test
    phi_arr = np.concatenate((phi_train, phi_test), axis=0)
    return phi_arr


def compute_pnml_logloss(phi_arr: np.ndarray, y_gt: np.ndarray, theta_genies: np.ndarray, var_list: list,
                         nfs: np.ndarray) -> float:
    var_list = np.array(var_list)
    y_hat = np.array([x @ theta_genie for x, theta_genie in zip(phi_arr, theta_genies)]).squeeze()
    prob = np.exp(-(y_hat - y_gt) ** 2 / (2 * var_list)) / np.sqrt(2 * np.pi * var_list)

    # Normalize by the pnml normalization factor
    prob /= nfs
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss


class Pnml:
    def __init__(self, phi_train: np.ndarray, y_train: np.ndarray, lamb: float = 0.0,
                 is_y_one_sided_interval: bool = True):

        # The interval for possible y, for creating pdf
        self.is_y_one_sided_interval = is_y_one_sided_interval
        self.y_to_eval = np.append(0, np.logspace(-16, 5, 1000))
        if self.is_y_one_sided_interval is False:
            self.y_to_eval = np.unique(np.append(self.y_to_eval, -self.y_to_eval))

        # Train feature matrix and labels
        self.phi_train = phi_train
        self.y_train = y_train

        # Regularization term
        self.lamb = lamb

        # ERM least squares parameters
        self.theta_erm = fit_least_squares_estimator(self.phi_train, self.y_train, lamb=self.lamb)

        # Indication of success or fail
        self.res_dict = None

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray):
        return fit_least_squares_estimator(phi_arr, y, lamb=self.lamb)

    def predict_erm(self, phi_test: np.ndarray) -> float:
        return float(self.theta_erm.T @ phi_test)

    def calc_norm_factor(self, phi_test: np.array, var: float = None) -> float:
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features to evaluate.
        :param var: genie's variance.
        :return: log normalization factor.
        """
        y_vec = self.create_y_vec_to_eval(phi_test, self.theta_erm, self.y_to_eval)
        thetas = self.calc_genie_thetas(phi_test, y_vec)

        # Calc genies predictions
        probs_of_genies = self.calc_probs_of_genies(phi_test, y_vec, thetas, var)

        # Integrate to find the pNML normalization factor
        norm_factor = float(np.trapz(probs_of_genies, x=y_vec))

        if self.is_y_one_sided_interval is True:
            norm_factor = 2 * norm_factor

        res_dict = self.verify_empirical_pnml_results(norm_factor, probs_of_genies)
        self.res_dict = res_dict
        return norm_factor

    @staticmethod
    def verify_empirical_pnml_results(norm_factor: float, probs_of_genies: np.ndarray) -> dict:
        res_dict = {'message': '', 'success': True}

        # Some check for converges:
        if norm_factor < 1.0:
            # Expected positive regret
            res_dict['message'] += 'Negative regret={:.3f} '.format(np.log(norm_factor))
            res_dict['success'] = False
        if probs_of_genies[-1] > np.finfo('float').eps:
            # Expected probability 0 at the edges
            res_dict['message'] += 'Interval is too small prob={}. '.format(probs_of_genies)
            res_dict['success'] = False
        return res_dict

    @staticmethod
    def create_y_vec_to_eval(phi_test: np.ndarray, theta_erm: np.ndarray, y_to_eval: np.ndarray) -> np.ndarray:
        """
        Adapt the y interval to the test sample.
        we want to predict around the ERM prediction based on the analytical result.
        :param phi_test: the test sample data.
        :param theta_erm: the erm parameters
        :param y_to_eval: the basic y interval
        :return: the shifted y interval based on the erm prediction
        """
        y_pred = theta_erm.T @ phi_test
        y_vec = y_to_eval + y_pred
        return y_vec

    def calc_genie_thetas(self, phi_test: np.ndarray, y_vec: np.ndarray) -> list:
        phi_arr = add_test_to_train(self.phi_train, phi_test)
        thetas = [self.fit_least_squares_estimator(phi_arr, np.append(self.y_train, y)) for y in y_vec]
        return thetas

    @staticmethod
    def calc_probs_of_genies(phi_test, y_trained: np.ndarray, thetas: np.ndarray, var: float) -> np.ndarray:
        """
        Calculate the genie probability of the label it was trained with
        :param phi_test: test set sample
        :param y_trained: The labels that the genie was trained with
        :param thetas: The fitted parameters to the label (the trained genie)
        :param var: the variance (sigma^2)
        :return: the genie probability of the label it was trained with
        """
        y_hat = np.array([theta.T @ phi_test for theta in thetas]).squeeze()
        y_trained = y_trained.squeeze()
        probs_of_genies = np.exp(-(y_trained - y_hat) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        return probs_of_genies


class PnmlMinNorm(Pnml):
    def __init__(self, constrain_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.lamb == 0.0

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor
        self.max_norm = self.constrain_factor * calc_theta_norm(self.theta_erm)

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray) -> np.ndarray:
        max_norm = self.max_norm
        theta = fit_norm_constrained_least_squares(phi_arr, y, max_norm)
        return theta


