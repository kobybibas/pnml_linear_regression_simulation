import numpy as np

from loguru import logger


class DataPolynomial:

    def __init__(self, x_train, y_train):
        """
        Create the training data.
        :param params: parameters for creating the training data.
        """

        # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x)
        self.phi_train = None

        # The least squares empirical risk minimization solution
        self.theta_erm = None

        # Generate the training data.
        self.x = np.array(x_train)
        self.y = np.array(y_train)

    def create_train(self, poly_degree: int):
        """
        Convert data points to feature matrix: phi=[x0^0,x0^1,x0^2...;x1^0,x1^1,x1^2...;x2^0,x2^1,x2^2...]
        :param poly_degree: the assumed polynomial degree of the training set.
        :return: phi: training set feature matrix.
        """

        # Create Feature matrix
        logger.info('Create train: num of features {}'.format(poly_degree))
        self.phi_train = self.convert_to_features(poly_degree)
        return self.phi_train

    def convert_to_features(self, pol_degree: int) -> np.ndarray:
        """
        Convert the training point to feature matrix.
        :param pol_degree: the assumed polynomial degree of the data.
        :return: phy = [x0^0 , x0^1, ... , x0^pol_degree; x1^0 , x1^1, ... , x1^pol_degree].T
        """
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(self.x, n))
        phi = np.asarray(phi)
        return phi

    @staticmethod
    def convert_point_to_features(x: float, pol_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param pol_degree: the assumed polynomial degree.
        :return: phi = [x^0,x^1,x^2,...].T
        """
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(x, n))
        phi = np.asarray(phi)

        if len(phi.shape) == 1:
            phi = phi[:, np.newaxis]

        return phi

    def get_data_points_as_list(self):
        """
        :return: list of training set data. list of training set labels.
        """
        return self.x.tolist(), self.y.tolist()

    def get_labels_array(self) -> np.ndarray:
        """
        :return: the labels of the training set.
        """
        return self.y

    @staticmethod
    def fit_least_squares_estimator(phi: np.ndarray, y: np.ndarray, lamb: float = 0.0) -> np.ndarray:
        """
        Fit ERM least squares estimator
        :param phi: the training set features matrix.
        :param y: the labels vector.
        :param lamb: regularization term.
        :return: the fitted parameters.
        """
        phi_phi_t = phi.dot(phi.T)
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        theta = phi_phi_t_inv.dot(phi).dot(y)
        return theta


class DataCosine(DataPolynomial):

    def convert_to_features(self, max_freq: int) -> np.ndarray:
        """
        Convert the training point to feature matrix.
        :param pol_degree: the assumed polynomial degree of the data.
        :return: phy = [x0^0 , x0^1, ... , x0^pol_degree; x1^0 , x1^1, ... , x1^pol_degree].T
        """
        phi = []
        for n in range(max_freq + 1):
            if n == 0:
                phi.append(np.array([1] * len(self.x)))
            elif n % 2 == 0:  # Even
                phi.append(np.cos(2 * np.pi * self.x / n))
            else:  # Odd
                phi.append(np.sin(2 * np.pi * self.x / n))
        phi = np.asarray(phi)
        return phi

    @staticmethod
    def convert_point_to_features(x: np.ndarray, max_freq: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param max_freq: proportional to the maximum frequency.
        :return: phi = [x^0,x^1,x^2,...].T
        """
        phi = []
        for n in range(max_freq + 1):
            if n == 0:
                phi.append(1)
            elif n % 2 == 0:  # Even
                phi.append(np.cos(2 * np.pi * x / n))
            else:  # Odd
                phi.append(np.sin(2 * np.pi * x / n))
        phi = np.asarray(phi)
        if len(phi.shape) == 1:
            phi = phi[:, np.newaxis]

        return phi
