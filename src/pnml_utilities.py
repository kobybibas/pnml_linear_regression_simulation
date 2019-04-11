import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

from data_utilities import DataPolynomial


class PNMLParameters:
    def __init__(self):
        # The interval for possible y
        self.y_max = 200.0
        self.y_min = -200.0
        self.dy = 0.01

        # Low bound of the empirical variance
        self.min_sigma = 0.01

    def __str__(self):
        string = 'PNMLParameters:\n'
        string += '    y_max: {}\n'.format(self.y_max)
        string += '    y_min: {}\n'.format(self.y_min)
        string += '    dy: {}\n'.format(self.dy)
        string += '    min_sigma: {}\n'.format(self.min_sigma)
        return string


class Pnml:
    def __init__(self):
        self.pnml_params = PNMLParameters()

        # log normalization factor
        self.regret = None

        # The normalization factor
        self.k = None

        # The PDF of the pnml predictor at the test point x_test
        self.p_y = None

        # The interval for possible y, for creating pdf
        self.y_to_eval = None

        # Training points
        self.y_train = None
        self.x_train = None

        # Fitted least squares parameters
        self.theta_erm = None

        # Train feature matrix
        self.phi_train = None

        # Test feature vector
        self.phi_test = None

        # Train and test feature matrix
        self.phi = None

        # P_N matrix
        self.phi_phi_t_inv = None

    def get_y_interval(self, y_min: float = None, y_max: float = None, dy: float = None) -> list:
        """
        Build list on which the probability will be evaluated.
        :param y_min: the lower bound of the interval.
        :param y_max: the higher bound of the interval.
        :param dy: the difference between y ticks.
        :return: y list on which to eval the probability.
        """
        if self.y_to_eval is None:

            if y_max is None and dy is not None and y_min is None:
                y_max = self.pnml_params.y_max
                y_min = self.pnml_params.y_min
                dy = self.pnml_params.dy

            assert y_max is not None
            assert y_min is not None
            assert dy is not None

            # Initialize evaluation interval
            self.y_to_eval = np.arange(y_min, y_max, dy).round(3)
        return self.y_to_eval.tolist()

    def get_predictor(self):
        """
        :return: Probability density function in the test point of the labels (y).
        """
        if self.p_y is None:
            ValueError('p_y is none. First call to compute_predictor method')
        return self.p_y

    def compute_predictor(self, x_test: np.ndarray, pnml_params: PNMLParameters, data_h: DataPolynomial,
                          lamb: float):
        """
        Compute the pnml learner.
        :param x_test: training data.
        :param pnml_params: pnml learner parameters.
        :param data_h: data class handler.
        :param lamb: regularization term value
        :return:
        """

        # Assign training features
        assert isinstance(data_h.poly_degree, int)
        if data_h.phi_train is None:
            data_h.create_train(data_h.poly_degree)
        self.theta_erm = data_h.fit_least_squares_estimator(data_h.phi_train,
                                                            data_h.get_labels_array(),
                                                            lamb)

        self.phi_train = data_h.phi_train
        self.y_train = data_h.x
        self.x_train = data_h.y

        self.pnml_params = pnml_params
        self.phi_test = data_h.convert_point_to_features(x_test, data_h.poly_degree)
        self.phi = data_h.add_test_to_train(self.phi_train, self.phi_test)

        # Calculate the normalization factor
        self.k = self.calculate_norm_factor(self.phi, lamb)
        self.regret = np.log(self.k)

        # Calculate pnml estimator pdf
        self.p_y = self.create_pnml_pdf(self.get_y_interval())

    def calculate_norm_factor(self, phi: np.ndarray, lamb: float = 0.0) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi: the feature matrix (train+test).
        :param lamb: regularization term value.
        :return: normalization factor.
        """
        # Assuming the test features are last
        phi_phi_t = np.matmul(phi, phi.transpose())
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        k = 1 / (1 - self.phi_test.T.dot(phi_phi_t_inv).dot(self.phi_test))

        # For future calculations
        self.phi_phi_t_inv = phi_phi_t_inv
        return float(k)

    def create_pnml_pdf(self, y_to_eval: list):
        """
        Create Probability density function of phi_test
        :param y_to_eval: the labels on which the probability will be calculated.
        :return: probability list. p_y.sum() should be 1.0
        """

        # Computing the estimation std
        y_hat_train = self.phi_train.T.dot(self.theta_erm)
        sigma = (np.array(self.y_train) - y_hat_train).std()
        sigma = np.max([self.pnml_params.min_sigma, sigma])

        # Compute the mean
        y_hat_test = self.phi_test.T.dot(self.theta_erm)

        # Create the distribution
        rv = norm(loc=float(y_hat_test), scale=float(sigma * self.k))

        # The probability p(y_n|x_n,z^N)
        p_y = rv.pdf(y_to_eval)
        return p_y


class PnmlNoSigma(Pnml):
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
    def argmax_q_y(y_test, x_test, x_train, y_train, P_N, theta, c_0):
        sigma_2 = PnmlNoSigma.calc_sigma_2(x_test, y_test, x_train, y_train, P_N, theta, c_0)
        exponent = np.exp(-np.power(y_test - x_test.T.dot(theta), 2) / (2 * sigma_2 * np.power(c_0, 2)))
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


def get_argmax_prediction(exp_df_dict: dict) -> dict:
    """
    Get the argument of the maximum probability
    :param exp_df_dict: {key:  columns: x test. iloc: pdf on y}
    :return: the prediction at test point
    """
    # exp_df: columns: x test. iloc: pdf on y

    argmax_prediction_dict = {}
    for key, exp_df in exp_df_dict.items():
        argmax_prediction_dict[key] = exp_df.idxmax(axis=0).tolist()
    return argmax_prediction_dict


def get_mean_prediction(exp_df_dict):
    """
    Get the argument of the mean probability
    :param exp_df_dict: {key:  columns: x test. iloc: pdf on y}
    :return: the prediction at test point
    """
    # exp_df: columns: x test. iloc: pdf on y

    mean_prediction_dict = {}
    for key, exp_df in exp_df_dict.items():
        y_values = exp_df.index.values
        dy = abs((y_values[:-1] - y_values[1:]).mean())
        mean_prediction_dict[key] = []
        for x_point, prob in exp_df.iteritems():
            mean_prediction_dict[key].append((y_values * prob * dy).sum())
    return mean_prediction_dict


def twice_universality(twice_universal_dict: dict):
    """
    Calculating the twice universality predictor.
    :param twice_universal_dict:
    :return:
    """
    model_families = list(twice_universal_dict.keys())
    x_test = twice_universal_dict[model_families[0]].columns
    y_values = twice_universal_dict[model_families[0]].index.values
    dy = abs(y_values[:-1] - y_values[1:]).mean()
    # Initialize dataframe per test sample
    twice_df_single = pd.DataFrame(columns=model_families, index=y_values)

    # Initialize output data frame
    twice_df = pd.DataFrame(columns=x_test, index=y_values)
    for x in x_test:
        for key in model_families:
            twice_df_single[key] = twice_universal_dict[key][x]
        max_prob = twice_df_single.max(axis=1)
        twice_df[x] = max_prob / (dy * max_prob).sum()
    return twice_df
