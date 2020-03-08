import numpy as np
from loguru import logger
from scipy.optimize import NonlinearConstraint, minimize

from pnml_utils import Pnml


class PnmlMinNorm(Pnml):

    def calc_constraint_theta(self, phi: np.ndarray, y: np.ndarray, max_norm: float):

        def cons_f(x):
            return (x ** 2).sum()

        def cons_J(x):
            return 2 * x

        def mse(x):
            return np.power(y - x.T.dot(phi), 2).sum()

        def cons_H(x, v):
            return 2 * np.eye(x.shape[0])

        x0 = self.theta_erm

        nonlinear_constraint = NonlinearConstraint(cons_f, 0, max_norm, jac=cons_J, hess=cons_H)
        res = minimize(mse, x0, constraints=[nonlinear_constraint],  # method='trust-constr',
                       options={'disp': False, 'maxiter': 120})
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

    def argmax_q_y(self, y_test, phi_test, phi, y_train, sigma_2, max_norm):
        """

        :param y_test: Test sample label
        :param phi_test: Test sample data
        :param phi: features of all dataset, train + test sample
        :param y_train: trainset labels
        :param sigma_2: noise variance
        :param max_norm: norm constraint of the least squares predictor
        :return:
        """
        y = np.append(y_train, y_test)
        theta = self.calc_constraint_theta(phi, y, max_norm)
        y_hat = phi_test.T.dot(theta)
        prob = (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(-np.power(y_test - y_hat, 2) / (2 * sigma_2))
        return prob

    def execute_regret_calc(self, phi_test: np.array, phi_train: np.array, y_train: np.array, lamb: float):
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features.
        :param phi_train: train features matrix.
        :param y_train: train labels
        :param lamb: regularization term value.
        :return: log normalization factor.
        """
        phi = self.add_test_to_train(phi_train, phi_test)

        # ERM min norm
        norm_constrain = np.sum(self.theta_erm ** 2)
        sigma_square = self.min_sigma_square

        # ERM prediction
        y_pred = self.theta_erm.T.dot(phi)[-1]

        # Numerical integration
        y_vec = self.y_to_eval + y_pred
        res = [self.argmax_q_y(y, phi_test, phi, y_train, sigma_square, norm_constrain) for y in y_vec]
        k = np.trapz(np.array(res).squeeze(), x=y_vec)

        regret = np.log(k + np.finfo(float).eps)
        return regret

    @staticmethod
    def min_norm_solution(phi, y):

        # Over parametrization
        if phi.shape[0] > phi.shape[1]:
            phi_phi_t_inv = np.linalg.inv(phi.T.dot(phi))
            theta = phi.dot(phi_phi_t_inv).dot(y)
        else:
            phi_phi_t_inv = np.linalg.inv(phi.dot(phi.T))
            theta = phi_phi_t_inv.dot(phi.T).dot(y)

        return theta
