import logging
import math

import numpy as np

from learner_classes.learner_utils import estimate_sigma_with_valset
from learner_classes.pnml_utils import add_test_to_train, compute_pnml_logloss
from learner_classes.pnml_utils import fit_least_squares_with_max_norm_constrain, fit_least_squares_estimator

logger = logging.getLogger(__name__)


class AnalyticalPNML:
    def __init__(self, phi_train, theta_erm):
        """
        :param phi_train: data matrix, each row corresponds to train sample
        :param theta_erm: the minimum norm estimator.
        """

        # Calc constants
        self.n, self.m = phi_train.shape
        self.u, self.h_square, uh = np.linalg.svd(phi_train.T @ phi_train, full_matrices=True)
        self.n_effective = self.calc_effective_trainset_size(self.h_square, self.n)
        self.is_overparam = not self.n > self.m

        # Learn-able parameters intermediate results
        self.theta_erm = theta_erm
        self.theta_erm_norm = np.mean(self.theta_erm ** 2)
        self.P_N = np.linalg.pinv(phi_train.T @ phi_train)
        self.theta_mn_P_N_theta_mn = theta_erm.T.dot(self.P_N).dot(theta_erm)

    @staticmethod
    def calc_effective_trainset_size(h_square: np.ndarray, n_trainset: int) -> int:
        """
        Calculate the effective dimension of the training set.
        :param h_square: the singular values of the trainset correlation matrix
        :param n_trainset: the training set size
        :return: The effective dim
        """
        n_effective = min(np.sum(h_square > np.finfo('float').eps), n_trainset)
        return n_effective

    def calc_over_param_norm_factor(self, phi_test: np.ndarray, sigma_square: float) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features. Column vector
        :param sigma_square: the variance of the noise.
        :return: The normalization factor.
        """

        nf0 = self.calc_under_param_norm_factor(phi_test)

        if self.is_overparam is True:
            # ||x_\bot||^2
            x_bot_square = np.sum(((self.u.T.dot(phi_test)) ** 2)[self.n:])

            # Over parametrization region
            nf = nf0 + (2 * math.gamma(7 / 6) / np.sqrt(np.pi)) * (
                    (self.theta_mn_P_N_theta_mn ** 2) * (x_bot_square ** 4) / (sigma_square ** 2)) ** (1 / 6)
        else:
            # Under parametrization region
            nf = nf0
        return nf

    def calc_under_param_norm_factor(self, phi_test: np.ndarray) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features.
        :return: normalization factor.
        """

        # x^T P_N^|| x
        n = min(np.sum(self.h_square > np.finfo('float').eps), self.n)
        x_P_N_x = np.sum((((self.u.T.dot(phi_test)).squeeze() ** 2)[:n] / self.h_square[:n]))
        nf = 1 + x_P_N_x
        return nf

    def fit_overparam_genie(self, phi_train, y_train, phi_test, y_test) -> np.ndarray:
        # Add train to test
        phi_arr = add_test_to_train(phi_train, phi_test)
        y = np.append(y_train, y_test)

        # Fit linear regression
        norm_constrain = self.theta_erm_norm
        theta_genie = fit_least_squares_with_max_norm_constrain(phi_arr, y, norm_constrain,
                                                                minimize_dict=None, theta_0=self.theta_erm)
        return theta_genie

    @staticmethod
    def fit_underparam_genie(phi_train, y_train, phi_test, y_test):

        # Add train to test
        phi_arr = add_test_to_train(phi_train, phi_test)
        y = np.append(y_train, y_test)

        assert phi_arr.shape[0] == len(y)

        # Fit linear regression
        theta_genie = fit_least_squares_estimator(phi_arr, y, lamb=0.0)
        return theta_genie


def calc_analytical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray,
                                     x_val: np.ndarray, y_val: np.ndarray,
                                     x_test: np.ndarray, y_test: np.ndarray,
                                     theta_erm: np.ndarray = None,
                                     theta_genies: list = None) -> (dict, dict, dict):
    # Fit ERM
    if theta_erm is None:
        theta_erm = fit_least_squares_estimator(x_train, y_train, lamb=0.0)
    var = estimate_sigma_with_valset(x_val, y_val, theta_erm)

    # Initialize output
    res_dict_pnml, res_dict_pnml_isit, res_dict_genie = {}, {}, {}
    num_features, trainset_size = x_train.shape

    # Fit genie
    pnml_h = AnalyticalPNML(x_train, theta_erm)
    if theta_genies is None:
        if trainset_size > num_features:
            # under param region
            theta_genies = [pnml_h.fit_underparam_genie(x_train, y_train, x.T, y) for x, y in zip(x_test, y_test)]
        else:
            # over param region
            theta_genies = [pnml_h.fit_overparam_genie(x_train, y_train, x.T, y) for x, y in zip(x_test, y_test)]

    # pNML
    norm_factors = np.array([pnml_h.calc_over_param_norm_factor(x.T, var) for x in x_test])
    res_dict_pnml['regret'] = np.log(norm_factors).mean()
    res_dict_pnml['test_logloss'] = compute_pnml_logloss(x_test, y_test, theta_genies, var, norm_factors)

    # pNML isit
    norm_factors = np.array([pnml_h.calc_under_param_norm_factor(x.T) for x in x_test])
    res_dict_pnml_isit['regret'] = np.log(norm_factors).mean()
    res_dict_pnml_isit['test_logloss'] = compute_pnml_logloss(x_test, y_test, theta_genies, var, norm_factors)

    # Genie
    res_dict_genie['test_logloss'] = compute_pnml_logloss(x_test, y_test, theta_genies, var, 1.0)
    res_dict_genie['theta_norm'] = np.mean(np.array(theta_genies) ** 2, axis=0).mean()

    return res_dict_pnml, res_dict_pnml_isit, res_dict_genie
