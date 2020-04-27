import logging

import numpy as np

logger = logging.getLogger(__name__)


class Pnml:
    def __init__(self, phi_train: np.ndarray, y_train: np.ndarray, lamb: float = 0.0, min_sigma_square: float = 1e-6):
        # Minimum variance for prediction
        self.min_sigma_square = min_sigma_square

        # The PDF of the pnml predictor at the test point x_test
        self.p_y = None

        # dict of the genies output:
        #     y_vec: labels that were evaluated, probs:probability of the corresponding labels
        self.genies_output = {'y_vec': None, 'probs': None}

        # The interval for possible y, for creating pdf
        self.y_to_eval = np.arange(-1000, 1000, 0.01)

        # Train feature matrix and labels
        self.phi_train = phi_train
        self.y_train = y_train

        # P_N matrix
        self.phi_phi_t_inv = None

        # Regularization term
        self.lamb = lamb

        # Fitted least squares parameters
        self.theta_erm = self.fit_least_squares_estimator(phi_train, self.y_train, self.lamb)

    def set_y_interval(self, y_min: float, y_max: float, dy: float, is_adaptive: bool = False):
        """
        Build list on which the probability will be evaluated.
        :param y_min: the lower bound of the interval.
        :param y_max: the higher bound of the interval.
        :param dy: the difference between y ticks.
        :param is_adaptive: gives more dy around zero.
        :return: y list on which to eval the probability.
        """
        # Initialize evaluation interval
        y_to_eval = np.arange(y_min, y_max, dy)
        if is_adaptive is True:
            y_to_eval = np.concatenate((np.arange(0, 0.001, 1e-7),
                                        np.arange(0.001, 1, 1e-3),
                                        np.arange(1.0, 10, 0.1),
                                        np.arange(10, 100, 1.0),
                                        np.arange(100, 1000, 10.0)))
            y_to_eval = np.unique(np.concatenate((-y_to_eval, y_to_eval)))
        logger.info('y_to_eval.shape: {}'.format(y_to_eval.shape))
        self.y_to_eval = y_to_eval

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

    def execute_regret_calc(self, phi_test: np.array):
        phi = self.add_test_to_train(self.phi_train, phi_test)

        # Calculate the normalization factor
        k = self.calculate_norm_factor(phi_test, phi, self.lamb)
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

    # def create_pnml_pdf(self, y_to_eval: list):
    #     """
    #     Create Probability density function of phi_test
    #     :param y_to_eval: the labels on which the probability will be calculated.
    #     :return: probability list. p_y.sum() should be 1.0
    #     """
    #
    #     # Computing the estimation std
    #     y_hat_train = self.phi_train.T.dot(self.theta_erm)
    #     sigma = (np.array(self.y_train) - y_hat_train).std()
    #     sigma = np.max([self.pnml_params.min_sigma, sigma])
    #
    #     # Compute the mean
    #     y_hat_test = self.phi_test.T.dot(self.theta_erm)
    #
    #     # Create the distribution
    #     rv = norm(loc=float(y_hat_test), scale=float(sigma * self.k))
    #
    #     # The probability p(y_n|x_n,z^N)
    #     p_y = rv.pdf(y_to_eval)
    #     return p_y

    def fit_least_squares_estimator(self, phi: np.ndarray, y: np.ndarray, lamb: float = 0.0) -> np.ndarray:
        """
        Fit least squares estimator
        :param phi: the training set features matrix.
        :param y: the labels vector.
        :param lamb: regularization term.
        :return: the fitted parameters.
        """
        phi_phi_t = phi.dot(phi.T)
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        theta = phi_phi_t_inv.dot(phi).dot(y)
        return theta

    def predict_erm(self, phi_test: np.ndarray) -> float:
        return float(self.theta_erm.T.dot(phi_test))
