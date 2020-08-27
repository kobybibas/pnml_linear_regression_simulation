import logging
import math

import numpy as np
import pandas as pd

from learner_utils.pnml_utils import compute_pnml_logloss

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

    def calc_norm_factor_with_lambda(self, phi_test: np.ndarray, lamb: float) -> float:
        """
        Calculate the normalization factor of the pnml learner with regularization.
        :param phi_test: test features.
        :param lamb: the regularization factor
        :return: normalization factor.
        """

        # x^T P_N^|| x
        x_P_N_x = np.sum((((self.u.T.dot(phi_test)).squeeze() ** 2) / (self.h_square + lamb)))
        nf = 1 + x_P_N_x
        return nf


def calc_analytical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                                     theta_erm: np.ndarray, theta_genies: list, var_genies: list) -> pd.DataFrame:
    # Fit genie
    pnml_h = AnalyticalPNML(x_train, theta_erm)

    # pNML
    norm_factors = np.array([pnml_h.calc_over_param_norm_factor(x.T, var) for x, var in zip(x_test, var_genies)])
    res_dict_pnml = {
        'analytical_pnml_regret': np.log(norm_factors),
        'analytical_pnml_test_logloss': compute_pnml_logloss(x_test, y_test, theta_genies, var_genies, norm_factors)}
    analytical_pnml_df = pd.DataFrame(res_dict_pnml)
    return analytical_pnml_df
