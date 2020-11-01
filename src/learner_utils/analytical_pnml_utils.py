import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnalyticalPNML:
    def __init__(self, phi_train, theta_erm, eigenvalue_threshold: float = 1e-12):
        """
        :param phi_train: data matrix, each row corresponds to train sample
        :param theta_erm: the minimum norm estimator.
        """

        # Calc constants
        self.n, self.m = phi_train.shape
        self.u, self.h_square, _ = np.linalg.svd(phi_train.T @ phi_train, full_matrices=True)
        self.n_effective = self.calc_effective_trainset_size(self.h_square, self.n, eigenvalue_threshold)
        self.is_overparam = self.m >= self.n
        if self.n != self.n_effective:
            logger.info('The effective rank is different: [n n_effective]=[{} {}]'.format(self.n, self.n_effective))

        # Learn-able parameters intermediate results
        self.theta_mn_P_N_theta_mn = self.calc_trainset_subspace_projection(theta_erm)

        self.nf0, self.nf1, self.nf2, self.x_bot_square = 0, 0, 0, 0

    @staticmethod
    def calc_effective_trainset_size(h_square: np.ndarray, n_trainset: int, eigenvalue_threshold: float) -> int:
        """
        Calculate the effective dimension of the training set.
        :param h_square: the singular values of the trainset correlation matrix
        :param n_trainset: the training set size
        :param eigenvalue_threshold: the smallest eigenvalue that is allowed
        :return: The effective dim
        """
        n_effective = min(np.sum(h_square > eigenvalue_threshold), n_trainset)
        return int(n_effective)

    def calc_norm_factor(self, phi_test: np.ndarray, sigma_square: float) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features. Column vector
        :param sigma_square: the variance of the noise.
        :return: The normalization factor.
        """
        # Initialize
        self.nf0, self.nf1, self.nf2, self.x_bot_square = 0, 0, 0, 0

        # Under param
        self.nf0 = float(self.calc_under_param_norm_factor(phi_test))

        # Over param
        if self.is_overparam is True:
            # ||x_\bot||^2
            x = phi_test
            self.x_bot_square = self.calc_x_bot_square(x)
            x_parallel_square = self.calc_trainset_subspace_projection(x)
            self.nf1 = float(2 * self.x_bot_square * x_parallel_square)

            c = self.x_bot_square * self.theta_mn_P_N_theta_mn / (np.pi * sigma_square)
            if c < 0:
                logger.warning('Lower than zero. x_bot_square={} theta_mn_P_N_theta_mn={}'.format(
                    self.x_bot_square, self.theta_mn_P_N_theta_mn))
            self.nf2 = float(3 * np.power(c, 1. / 3))

        nf = self.nf0 + self.nf1 + self.nf2
        return float(nf)

    def calc_x_bot_square(self, x: np.ndarray) -> float:
        x_bot_square = np.sum(np.power(self.u.T @ x, 2)[self.n_effective:])
        return float(x_bot_square)

    def calc_trainset_subspace_projection(self, x: np.ndarray) -> float:
        x_projection = np.squeeze(np.power(self.u.T @ x, 2))[:self.n_effective]
        x_parallel_square = np.sum(x_projection / self.h_square[:self.n_effective])
        return float(x_parallel_square)

    def calc_under_param_norm_factor(self, phi_test: np.ndarray) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features.
        :return: normalization factor.
        """
        # x^T P_N^|| x
        x_P_N_x = self.calc_trainset_subspace_projection(phi_test)
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
                                     theta_erm: np.ndarray, variances: list) -> pd.DataFrame:
    # Fit genie
    pnml_h = AnalyticalPNML(x_train, theta_erm)

    # pNML
    norm_factors = np.array([pnml_h.calc_norm_factor(x.T, var) for x, var in zip(x_test, variances)])
    res_dict_pnml = {'analytical_pnml_regret': np.log(norm_factors)}
    analytical_pnml_df = pd.DataFrame(res_dict_pnml)
    return analytical_pnml_df
