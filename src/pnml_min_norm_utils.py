import logging
import math

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from pnml_utils import Pnml, add_test_to_train

logger = logging.getLogger(__name__)


class PnmlMinNorm(Pnml):
    def __init__(self, constrain_factor, *args, **kargs):
        super().__init__(*args, **kargs)

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

        #  The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        theta_erm_min_norm = super().fit_least_squares_estimator(self.phi_train, self.y_train)
        self.max_norm = self.constrain_factor * np.sum(theta_erm_min_norm ** 2)

        # Fitted least squares parameters with norm constraint
        self.fit_least_squares_estimator = self.fit_least_squares_estimator_with_max_norm
        self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.y_train, lamb=0.0)

    def set_constrain_factor(self, constrain_factor: float):
        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

    def fit_least_squares_estimator_with_max_norm(self, phi: np.ndarray, y: np.ndarray,
                                                  lamb: float = 0.0) -> np.ndarray:
        max_norm = self.max_norm

        def cons_f(x):
            return (x ** 2).sum()

        def cons_J(x):
            return 2 * x

        def mse(x):
            return np.power(y - x.T.dot(phi), 2).mean()

        x0 = self.theta_erm if self.theta_erm is not None else np.zeros(phi.shape[0])

        nonlinear_constraint = NonlinearConstraint(cons_f, 0, max_norm, jac=cons_J)
        res = minimize(mse, x0, constraints=[nonlinear_constraint], method='SLSQP',
                       options={'disp': False, 'maxiter': 1000, 'ftol': 1e-12})
        if res.success is False:
            logger.info('fit_least_squares_estimator_with_max_norm: Failed')
            logger.info('y: {}'.format(y[-1]))
            logger.info(res)
        theta = res.x
        return theta

    def calc_pnml_epsilon(self, y_test, phi_test, phi, y_train) -> float:
        """
        :param y_test: Test sample label
        :param phi_test: Test sample data
        :param phi: features of all dataset, train + test sample
        :param y_train: trainset labels
        :return:
        """
        y = np.append(y_train, y_test)
        theta = self.fit_least_squares_estimator(phi, y, lamb=0.0)
        y_hat = phi_test.T.dot(theta)
        return float(y_test - y_hat)

    @staticmethod
    def calc_genies_probs(epsilons: np.ndarray, sigma_square: float) -> np.ndarray:
        genies_probs = (1 / np.sqrt(2 * np.pi * sigma_square)) * np.exp(-epsilons ** 2 / (2 * sigma_square))
        return genies_probs

    def execute_regret_calc(self, phi_test: np.array):
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features to evaluate.
        :return: log normalization factor.
        """
        assert self.phi_train is not None
        assert self.theta_erm is not None

        phi = add_test_to_train(self.phi_train, phi_test)

        # Predict around the ERM prediction
        y_pred = self.theta_erm.T.dot(phi)[-1]
        y_vec = self.y_to_eval + y_pred

        # Genies prediction
        epsilons = np.array([self.calc_pnml_epsilon(y, phi_test, phi, self.y_train) for y in y_vec])
        genies_probs = self.calc_genies_probs(epsilons, self.min_sigma_square)

        # Integrate to find the pNML normalization factor
        norm_factor = np.trapz(genies_probs, x=y_vec)

        # The genies predictors
        self.genies_output = {'y_vec': y_vec, 'probs': genies_probs, 'epsilons': epsilons}
        regret = np.log(norm_factor)
        return regret


def calc_analytic_norm_factor(phi_train: np.ndarray, phi_test: np.ndarray,
                              theta_mn: np.ndarray, sigma_square: float) -> float:
    """
    Calculate the normalization factor of the pnml learner.
    :param phi_train: the train feature matrix (train+test).
    :param phi_test: test features.
    :param sigma_square: the variance of the noise.
    :param theta_mn: minimum norm solution parameters
    :return: normalization factor.
    """
    M, N = phi_train.shape
    u, h_square, uh = np.linalg.svd(phi_train.dot(phi_train.T), full_matrices=True)
    P_N = np.linalg.pinv(phi_train.dot(phi_train.T))
    theta_mn_P_N_theta_mn = theta_mn.T.dot(P_N).dot(theta_mn)

    # ||x_\bot||^2
    x_bot_square = np.sum(((u.T.dot(phi_test)) ** 2)[N:])

    nf0 = calc_gamma_0_norm_factor(phi_train, phi_test)
    # Analytical normalization factor
    nf = nf0 + (2 * math.gamma(7 / 6) / np.sqrt(np.pi)) * (
            (theta_mn_P_N_theta_mn ** 2) * (x_bot_square ** 4) / (sigma_square ** 2)) ** (1 / 6)
    return nf


def calc_gamma_0_norm_factor(phi_train: np.ndarray, phi_test: np.ndarray) -> float:
    """
    Calculate the normalization factor of the pnml learner.
    :param phi_train: the train feature matrix (train+test).
    :param phi_test: test features.
    :return: normalization factor.
    """
    M, N = phi_train.shape
    u, h_square, uh = np.linalg.svd(phi_train.dot(phi_train.T), full_matrices=True)

    # x^T P_N^|| x
    x_P_N_x = np.sum(((u.T.dot(phi_test)).squeeze() ** 2 / h_square)[:N])

    # Analytical normalization factor
    nf = 1 + x_P_N_x
    return nf


def calc_x_bot_square(phi_train: np.ndarray, phi_test: np.ndarray) -> float:
    """
    Calculate the normalization factor of the pnml learner.
    :param phi_train: the train feature matrix (train+test).
    :param phi_test: test features.
    :return: normalization factor.
    """
    M, N = phi_train.shape
    u, h_square, uh = np.linalg.svd(phi_train.dot(phi_train.T), full_matrices=True)

    # ||x_\bot||^2
    x_bot_square = np.sum(((u.T.dot(phi_test)) ** 2)[N:])
    return x_bot_square


class AnalyticalMinNormPNML:
    def __init__(self, phi_train, theta_mn):
        """
        :param phi_train: data matrix, each row corresponds to train sample
        :param theta_mn: the minimum norm estimator.
        """

        # Calc constants
        self.N, self.M = phi_train.shape
        self.u, self.h_square, uh = np.linalg.svd(phi_train.T @ phi_train, full_matrices=True)

        self.P_N = np.linalg.pinv(phi_train.T @ phi_train)
        self.theta_mn_P_N_theta_mn = theta_mn.T.dot(self.P_N).dot(theta_mn)

        self.is_overparam = self.M >= self.N

    def calc_over_param_norm_factor(self, phi_test: np.ndarray, sigma_square: float) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features. Column vector
        :param sigma_square: the variance of the noise.
        :return: normalization factor.
        """

        nf0 = self.calc_under_param_norm_factor(phi_test)

        if self.is_overparam is True:
            # ||x_\bot||^2
            x_bot_square = np.sum(((self.u.T.dot(phi_test)) ** 2)[self.N:])

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
        x_P_N_x = np.sum((((self.u.T.dot(phi_test)).squeeze() ** 2)[:self.N] / self.h_square[:self.N]))
        nf0 = 1 + x_P_N_x
        return nf0

    def fit_overparam_genie(self, theta_mn, phi_train, y_train, phi_test, y_test):

        # verify test is a column vector
        phi_test = np.expand_dims(phi_test, axis=1) if len(phi_test.shape) == 1 else phi_test

        # Add train to test
        phi = np.concatenate((phi_train, phi_test.T), axis=0)
        y = np.append(y_train, y_test)

        norm_constrain = np.mean(theta_mn ** 2)

        # x is the learn-able parameters
        def cons_f(x):
            return (x ** 2).mean()

        def cons_J(x):
            return 2 * x / x.shape[0]

        def mse(x):
            return np.power(y - phi @ x, 2).mean()

        x0 = theta_mn

        nonlinear_constraint = NonlinearConstraint(cons_f, 0, norm_constrain, jac=cons_J)
        res = minimize(mse, x0, constraints=[nonlinear_constraint], method='SLSQP',
                       options={'disp': False, 'maxiter': 100, 'ftol': 1e-6})
        if res.success is False:
            logger.info('fit_least_squares_estimator_with_max_norm: Failed')
            logger.info('y: {}'.format(y[-1]))
            logger.info(res)
        theta_genie = res.x
        return theta_genie

    def fit_underparam_genie(self, phi_train, y_train, phi_test, y_test):

        # verify test is a column vector
        phi_test = np.expand_dims(phi_test, axis=1) if len(phi_test.shape) == 1 else phi_test

        # Add train to test
        phi = np.concatenate((phi_train, phi_test.T), axis=0)
        y = np.append(y_train, y_test)

        # Fit linear regression
        inv = np.linalg.pinv(phi.T @ phi)
        theta_genie = inv @ phi.T @ y
        return theta_genie
