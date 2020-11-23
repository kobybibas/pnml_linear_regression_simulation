import logging

import numpy as np
import pandas as pd

from learner_utils.learner_helpers import calc_effective_trainset_size

logger = logging.getLogger(__name__)


class AnalyticalPNML:
    def __init__(self, phi_train, theta_erm, eigenvalue_threshold: float = np.finfo('float').eps):
        """
        :param phi_train: data matrix, each row corresponds to train sample
        :param theta_erm: the minimum norm estimator.
        """

        # Calc constants
        self.n, self.m = phi_train.shape
        self.u, self.h_square, _ = np.linalg.svd(phi_train.T @ phi_train, full_matrices=True)
        self.rank = min(self.n, self.m)
        self.rank_effective = calc_effective_trainset_size(self.h_square, self.rank, eigenvalue_threshold)
        self.is_overparam = self.m >= self.n
        if self.rank != self.rank_effective:
            logger.info('The effective rank is different: [rank effective]=[{} {}]. [m n]=[{} {}]'.format(
                self.rank, self.rank_effective, self.n, self.m))

        # Learn-able parameters intermediate results
        self.theta_mn_P_N_theta_mn = self.calc_trainset_subspace_projection(theta_erm)

        self.nf0, self.nf1, self.nf2, self.x_bot_square = 0, 0, 0, 0

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

        # ||x_\bot||^2
        x = phi_test
        self.x_bot_square = self.calc_x_bot_square(x)

        # Over param
        if self.is_overparam is True and self.x_bot_square > 1e-6:  # todo get as input
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
        x_bot_square = np.sum(np.power(self.u.T @ x, 2)[self.rank_effective:])
        return float(x_bot_square)

    def calc_trainset_subspace_projection(self, x: np.ndarray) -> float:
        x_projection = np.squeeze(np.power(self.u.T @ x, 2))[:self.rank_effective]
        x_parallel_square = np.sum(x_projection / self.h_square[:self.rank_effective])
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
