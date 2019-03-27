import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from data_utilities import DataPolynomial


class PNMLParameters:
    def __init__(self):
        # The interval for possible y
        self.y_max = 200.0
        self.dy = 0.01
        self.min_sigma = 0.01

    def __str__(self):
        string = 'PNMLParameters:\n'
        string += '    y_max: {}\n'.format(self.y_max)
        string += '    dy: {}\n'.format(self.dy)
        string += '    min_sigma: {}\n'.format(self.min_sigma)
        return string


class Pnml:
    def __init__(self):
        self.pnml_params = None

        # log normalization factor
        self.regret = None

        # The normalization factor
        self.c = None

        # The PDF of the pnml predictor at the test point x_test
        self.p_y = None

        # The interval for possible y, for creating pdf
        self.y_to_eval = None

        # Training variables
        self.y_train = None
        self.x_train = None
        self.theta_erm = None
        self.phi_train = None
        self.phi_test = None
        self.phi = None

        # P_N matrix
        self.phi_phi_t_inv = None

    def get_y_interval(self, y_max: int = None, dy: int = None):
        if self.y_to_eval is None:

            if y_max is None and dy is not None:
                y_max = self.pnml_params.y_max
                dy = self.pnml_params.dy

            assert y_max is not None
            assert dy is not None

            # Initialize evaluation interval
            self.y_to_eval = np.arange(-y_max, y_max, dy).round(3)
        return self.y_to_eval.tolist()

    def get_predictor(self):
        if self.p_y is None:
            ValueError('p_y is none. First call to compute_predictor method')
        return self.p_y

    def compute_predictor(self, x_test, pnml_params: PNMLParameters, data_h: DataPolynomial):

        # Assign training features
        assert isinstance(data_h.poly_degree, int)
        if data_h.theta_erm is None or data_h.phi_train is None:
            data_h.create_train(data_h.poly_degree)
        self.theta_erm = data_h.theta_erm
        self.phi_train = data_h.phi_train
        self.y_train = data_h.x
        self.x_train = data_h.y

        self.pnml_params = pnml_params
        self.phi_test = data_h.convert_point_to_features(x_test, data_h.poly_degree)
        self.phi = data_h.add_test_to_train(self.phi_train, self.phi_test)

        # Calculate the normalization factor
        self.c = self.calculate_norm_factor(self.phi, data_h.poly_degree)
        self.regret = np.log(self.c)

        # Calculate pnml estimator pdf
        self.p_y = self.create_pnml_pdf(self.get_y_interval())

    def calculate_norm_factor(self, phi: np.ndarray, lamb=0.0):
        # Assuming the test features are last

        phi_phi_t = np.matmul(phi, phi.transpose())
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        c = 1 / (1 - self.phi_test.T.dot(phi_phi_t_inv).dot(self.phi_test))

        # For future calculations
        self.phi_phi_t_inv = phi_phi_t_inv
        return float(c)

    def create_pnml_pdf(self, y_to_eval):

        # Computing the estimation std
        y_hat_train = self.phi_train.T.dot(self.theta_erm)
        sigma = (np.array(self.y_train) - y_hat_train).std()
        sigma = np.max([self.pnml_params.min_sigma, sigma])

        # Compute the mean
        y_hat_test = self.phi_test.T.dot(self.theta_erm)

        # Create the distribution
        rv = norm(loc=float(y_hat_test), scale=float(sigma * self.c))

        # The probability p(y_n|x_n,z^N)
        p_y = rv.pdf(y_to_eval)
        return p_y


class PnmlNoSigma(Pnml):
    def __init__(self):
        super().__init__()
        self.c_0 = None

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
    def argmax_q_y(y_test, x_test, x_train, y_train, P_N, theta, c_0):
        sigma_2 = PnmlNoSigma.calc_sigma_2(x_test, y_test, x_train, y_train, P_N, theta, c_0)
        exponent = np.exp(-np.power(y_test - x_test.T.dot(theta), 2) / (2 * sigma_2 * np.power(c_0, 2)))
        mul1 = 1 / np.sqrt(2 * np.pi * sigma_2)

        return mul1 * exponent

    def calculate_norm_factor(self, phi: np.ndarray, lamb=0.001):

        # Calculate the original pNML normalization factor
        self.c_0 = super().calculate_norm_factor(phi, lamb)

        # Calculate normalization factor using numerical integration
        c, _ = quad(self.argmax_q_y,
                    self.y_to_eval[0],
                    self.y_to_eval[-1],
                    args=(self.phi_test, self.phi_train, self.y_train,
                          self.phi_phi_t_inv, self.theta_erm, self.c_0))
        return c

    def create_pnml_pdf(self, y_to_eval):
        p_y = []
        for y_single in y_to_eval:
            p_y.append(self.argmax_q_y(y_single, self.phi_test, self.phi_train, self.y_train,
                                       self.phi_phi_t_inv, self.theta_erm, self.c_0))
        p_y = np.array(p_y) / self.c
        # print('p_y sum = ', p_y.sum() * self.pnml_params.dy)
        return p_y.squeeze()
