import logging

import numpy as np
import numpy.linalg as npl
import pandas as pd

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
        self.is_overparam = self.m >= self.n

        X = phi_train
        self.X_inv = X.T @ npl.inv(X @ X.T) if self.is_overparam else npl.inv(X.T @ X) @ X.T
        self.P_N = self.X_inv @ self.X_inv.T
        self.P_bot = np.eye(self.m) - self.X_inv @ X

        # Learn-able parameters intermediate results
        self.theta_erm = theta_erm
        self.theta_mn_P_N_theta_mn = theta_erm.T.dot(self.P_N).dot(theta_erm)

        self.nf0, self.nf1, self.nf2 = 0, 0, 0

    @staticmethod
    def calc_effective_trainset_size(h_square: np.ndarray, n_trainset: int) -> int:
        """
        Calculate the effective dimension of the training set.
        :param h_square: the singular values of the trainset correlation matrix
        :param n_trainset: the training set size
        :return: The effective dim
        """
        n_effective = min(np.sum(h_square > np.finfo('float').eps), n_trainset)
        return n_effective  # todo: how to use?

    def calc_norm_factor(self, phi_test: np.ndarray, sigma_square: float) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features. Column vector
        :param sigma_square: the variance of the noise.
        :return: The normalization factor.
        """
        # Initialize
        self.nf0, self.nf1, self.nf2 = 0, 0, 0

        # Under param
        self.nf0 = float(self.calc_under_param_norm_factor(phi_test))

        # Over param
        if self.is_overparam is True:
            # ||x_\bot||^2
            x = phi_test
            x_bot_square = x.T @ self.P_bot @ x

            self.nf1 = float(2 * x_bot_square * x.T @ self.P_N @ x)

            c = x_bot_square * self.theta_mn_P_N_theta_mn / (np.pi * sigma_square)
            if c < 0:
                logger.warning('Lower than zero. x_bot_square={} theta_mn_P_N_theta_mn={}'.format(
                    x_bot_square, self.theta_mn_P_N_theta_mn))
                logger.warning('P_bot={}'.format(self.P_bot))
            self.nf2 = float(3 * np.power(c, 1. / 3))

        nf = self.nf0 + self.nf1 + self.nf2
        return float(nf)

    def calc_under_param_norm_factor(self, phi_test: np.ndarray) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features.
        :return: normalization factor.
        """
        # x^T P_N^|| x
        nf = 1 + phi_test.T @ self.P_N @ phi_test
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
                                     theta_erm: np.ndarray, variances: list) -> pd.DataFrame:
    # Fit genie
    pnml_h = AnalyticalPNML(x_train, theta_erm)

    # pNML
    norm_factors = np.array([pnml_h.calc_norm_factor(x.T, var) for x, var in zip(x_test, variances)])
    res_dict_pnml = {'analytical_pnml_regret': np.log(norm_factors)}
    analytical_pnml_df = pd.DataFrame(res_dict_pnml)
    return analytical_pnml_df
