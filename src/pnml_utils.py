import numpy as np
from scipy.stats import norm


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

        # Minimum variance for prediction
        self.min_sigma_square = None

        # The PDF of the pnml predictor at the test point x_test
        self.p_y = None

        # The interval for possible y, for creating pdf
        self.y_to_eval = None

        # Training points
        self.y_train = None
        self.x_train = None
        self.y_train_vec = None

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

    def set_erm_predictor(self, theta_erm):
        self.theta_erm = theta_erm

    def set_y_interval(self, y_min: float, y_max: float, dy: float):
        """
        Build list on which the probability will be evaluated.
        :param y_min: the lower bound of the interval.
        :param y_max: the higher bound of the interval.
        :param dy: the difference between y ticks.
        :return: y list on which to eval the probability.
        """
        # Initialize evaluation interval
        self.y_to_eval = np.arange(y_min, y_max, dy)

    def get_predictor(self):
        """
        :return: Probability density function in the test point of the labels (y).
        """
        if self.p_y is None:
            ValueError('p_y is none. First call to compute_predictor method')
        return self.p_y

    @staticmethod
    def add_test_to_train(phi_train: np.ndarray, phi_test: np.ndarray) -> np.ndarray:
        """
        Add the test set feature to training set feature matrix
        :param phi_train: training set feature matrix.
        :param phi_test: test set feature.
        :return: concat train and test.
        """
        return np.concatenate((phi_train, phi_test), axis=1)

    def execute_regret_calc(self, phi_test: np.array, phi_train: np.array, y_train: np.array, lamb: float):
        phi = self.add_test_to_train(phi_train, phi_test)

        # Calculate the normalization factor
        k = self.calculate_norm_factor(phi_test, phi, lamb)
        regret = np.log(k + np.finfo(float).eps)
        return regret

    def calculate_norm_factor(self, phi_test: np.ndarray, phi: np.ndarray, lamb: float = 0.0) -> float:
        """
        Calculate the normalization factor of the pnml learner.
        :param phi_test: test features.
        :param phi: the feature matrix (train+test).
        :param lamb: regularization term value.
        :return: normalization factor.
        """
        # Assuming the test features are last
        phi_phi_t = np.matmul(phi, phi.transpose())
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        k = 1 / (1 - phi_test.T.dot(phi_phi_t_inv).dot(phi_test))

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

    @staticmethod
    def predict_erm(theta_erm: np.ndarray, phi_test: np.ndarray) -> float:
        return float(theta_erm.T.dot(phi_test))
