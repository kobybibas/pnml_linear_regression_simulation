import logging

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from pnml_utils import Pnml

logger = logging.getLogger(__name__)


class PnmlMinNorm(Pnml):
    def __init__(self, constrain_factor, *args, **kargs):
        super().__init__(*args, **kargs)

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor
        self.max_norm = self.calc_max_norm()

        # Fitted least squares parameters with norm constraint
        self.fit_least_squares_estimator = self.fit_least_squares_estimator_with_max_norm
        self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.y_train, lamb=0.0)

    def set_constrain_factor(self, constrain_factor: float):
        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

    def calc_max_norm(self) -> float:
        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        theta_erm_min_norm = super().fit_least_squares_estimator(self.phi_train, self.y_train)
        max_norm = self.constrain_factor * np.sum(theta_erm_min_norm ** 2)
        return max_norm

    def fit_least_squares_estimator_with_max_norm(self, phi: np.ndarray, y: np.ndarray,
                                                  lamb: float = 0.0) -> np.ndarray:
        max_norm = self.max_norm

        def cons_f(x):
            return (x ** 2).sum()

        def cons_J(x):
            return 2 * x

        def mse(x):
            return np.power(y - x.T.dot(phi), 2).sum()

        def cons_H(x, v):
            return 2 * np.eye(x.shape[0])

        x0 = self.theta_erm if self.theta_erm is not None else np.zeros(phi.shape[0])

        nonlinear_constraint = NonlinearConstraint(cons_f, 0, max_norm, jac=cons_J)  # , hess=cons_H)
        res = minimize(mse, x0, constraints=[nonlinear_constraint],  # method='trust-constr',
                       options={'disp': False, 'maxiter': 1000})
        theta = res.x

        if False:  # For debug
            # MSE on trainset
            mse = np.mean((theta.T.dot(phi)[:-1] - y[:-1]) ** 2)
            mse_erm = np.mean((x0.T.dot(phi)[:-1] - y[:-1]) ** 2)
            # Norm
            norm = np.sum(theta ** 2)
            norm_erm = np.sum(x0 ** 2)
            logger.info('ERM Genie. MSE=[{} {}] Norm=[{:.3f} {:.3f}]'.format(mse_erm, mse, norm_erm, norm))

        return theta

    def argmax_q_y(self, y_test, phi_test, phi, y_train, sigma_2):
        """

        :param y_test: Test sample label
        :param phi_test: Test sample data
        :param phi: features of all dataset, train + test sample
        :param y_train: trainset labels
        :param sigma_2: noise variance
        :return:
        """
        y = np.append(y_train, y_test)
        theta = self.fit_least_squares_estimator(phi, y, lamb=0.0)
        y_hat = phi_test.T.dot(theta)
        prob = (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(-np.power(y_test - y_hat, 2) / (2 * sigma_2))
        return float(prob)

    def calc_pnml_epsilon(self, y_test, phi_test, phi, y_train):
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

    def execute_regret_calc(self, phi_test: np.array):
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features to evaluate.
        :return: log normalization factor.
        """
        assert self.phi_train is not None
        assert self.theta_erm is not None

        sigma_2 = self.min_sigma_square
        phi = self.add_test_to_train(self.phi_train, phi_test)

        # ERM prediction
        y_pred = self.theta_erm.T.dot(phi)[-1]

        # Numerical integration
        y_vec = self.y_to_eval + y_pred
        epsilons = []
        for y in y_vec:
            epsilons.append(self.calc_pnml_epsilon(y, phi_test, phi, self.y_train))
        epsilons = np.array([epsilons])
        genies_probs_list = (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(-epsilons ** 2 / (2 * sigma_2))
        # genies_probs_list = [self.argmax_q_y(y, phi_test, phi, self.y_train, self.min_sigma_square) for y in y_vec]
        norm_factor = np.trapz(np.array(genies_probs_list).squeeze(), x=y_vec)

        # The genies predictors
        self.genies_output = {'y_vec': y_vec, 'probs': genies_probs_list, 'epsilons': epsilons}
        regret = np.log(norm_factor + np.finfo(float).eps)
        return regret
