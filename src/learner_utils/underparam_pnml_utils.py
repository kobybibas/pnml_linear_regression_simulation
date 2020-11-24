import logging

import numpy as np

from learner_utils.learner_helpers import fit_least_squares_estimator
from learner_utils.pnml_utils import BasePNML

logger = logging.getLogger(__name__)


class UnderparamPNML(BasePNML):

    def fit_least_squares_estimator(self, x_arr: np.ndarray, y_vec: np.ndarray, lamb: float) -> np.ndarray:
        return fit_least_squares_estimator(x_arr, y_vec, lamb=lamb)

    def optimize_var(self, x_test: np.ndarray, y_gt: float) -> float:
        nf = self.calc_norm_factor(x_test)
        var_best = float(np.power((y_gt - self.theta_erm.T @ x_test) / nf, 2))
        self.intermediate_dict['var_best'] = var_best
        self.var = float(var_best)
        return var_best

    def calc_norm_factor(self, x_test: np.array) -> float:
        """
        Calculate normalization factor using the analytical expression.
        :param x_test: test sample to evaluate.
        :return: log normalization factor.
        """
        # Verify it is a column vector
        rank = self.rank
        x_test = self.convert_to_column_vec(x_test)

        # Project on empirical correlation matrix and divide by the associated eigenvalue
        nf = 1 + np.sum(np.squeeze(self.u[:, :rank].T @ x_test) ** 2 / self.h[:rank] ** 2)
        self.intermediate_dict['nf'] = nf
        return nf

    def calc_pnml_logloss(self, x_test: np.ndarray, y_gt: float, nf: np.ndarray) -> float:
        # Make the input as column vector
        x_test = self.convert_to_column_vec(x_test)
        var = self.var

        # Add test to train
        logloss = 0.5 * np.log(2 * np.pi * var * (nf ** 2)) + (y_gt - self.theta_erm.T @ x_test) ** 2 / (
                2 * var * (nf ** 2))
        return float(logloss)

    def calc_genie_logloss(self, x_test: np.ndarray, y_gt: float, nf: float) -> float:
        # Make the input as column vector
        x_test = self.convert_to_column_vec(x_test)
        var = self.var

        # Add test to train
        logloss = 0.5 * np.log(2 * np.pi * var) + (y_gt - self.theta_erm.T @ x_test) ** 2 / (
                2 * var * (nf ** 2))
        return float(logloss)

    def verify_pnml_results(self) -> (bool, str):
        # Initialize output
        success, msg = True, ''
        nf = self.intermediate_dict['nf']

        # Some check for converges:
        if nf < 1.0 - np.finfo('float').eps:
            # Expected positive regret
            msg += 'Negative regret. norm_factor={:.6f} '.format(nf)
            success = False
        return success, msg

    def verify_var_results(self) -> (bool, str):
        # Initialize output
        success, msg = True, ''
        var_best = self.intermediate_dict['var_best']

        # Some check for converges:
        if np.isposinf(var_best):
            # Expected positive regret
            msg += 'Infinite var. Did not converge. '
            success = False
        return success, msg

    def calc_analytical_norm_factor(self, x_test):
        return self.calc_norm_factor(x_test)
