import numpy as np
from scipy.integrate import quad

from pnml_utilities import Pnml


class PnmlJoint(Pnml):
    def __init__(self):
        super().__init__()
        self.k_0 = None

    @staticmethod
    def calc_sigma_2(x_test: np.ndarray, y_test, x_train: np.ndarray, y_train: np.ndarray, P_N: np.ndarray,
                     theta: np.ndarray, c_0):
        summation = 0
        N = np.max(y_train.shape)
        y_hat_test = x_test.T.dot(theta)

        temp = P_N.dot(x_test).dot(y_test - y_hat_test)
        for i in range(N):
            x_i = x_train[:, i]
            y_i = y_train[i]
            y_hat_train = x_i.T.dot(theta)
            summation += np.power(y_i - y_hat_train - x_i.T.dot(temp), 2)

        return 1 / (N + 1) * (summation + np.power(y_test - y_hat_test, 2) / np.power(c_0, 2))

    @staticmethod
    def argmax_q_y(y_test, x_test, x_train, y_train, P_N, theta, k_0):
        sigma_2 = PnmlJoint.calc_sigma_2(x_test, y_test, x_train, y_train, P_N, theta, k_0)
        exponent = np.exp(-np.power(y_test - x_test.T.dot(theta), 2) / (2 * sigma_2 * np.power(k_0, 2)))
        mul1 = 1 / np.sqrt(2 * np.pi * sigma_2)

        return mul1 * exponent

    def calculate_norm_factor(self, phi: np.ndarray, lamb: float = 0.0):
        """
        Calculate the normalization factor of the pnml learner.
        :param phi: the feature matrix (train+test).
        :param lamb: regularization term value.
        :return: normalization factor.
        """

        # Calculate the original pNML normalization factor
        self.k_0 = super().calculate_norm_factor(phi, lamb)

        # Calculate normalization factor using numerical integration
        k, _ = quad(self.argmax_q_y,
                    self.y_to_eval[0],
                    self.y_to_eval[-1],
                    args=(self.phi_test, self.phi_train, self.y_train,
                          self.phi_phi_t_inv, self.theta_erm, self.k_0))
        return k

    def create_pnml_pdf(self, y_to_eval):
        """
        Create Probability density function of phi_test
        :param y_to_eval: the labels on which the probability will be calculated.
        :return: probability list. p_y.sum() should be 1.0
        """
        p_y = []
        for y_single in y_to_eval:
            p_y.append(self.argmax_q_y(y_single, self.phi_test, self.phi_train, self.y_train,
                                       self.phi_phi_t_inv, self.theta_erm, self.k_0))
        p_y = np.array(p_y) / self.k
        # print('p_y sum = ', p_y.sum() * self.pnml_params.dy)
        return p_y.squeeze()
