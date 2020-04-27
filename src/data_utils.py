import logging
from abc import abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class DataBase:
    def __init__(self, x_train: list, y_train: list, model_degree: int):
        # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x)
        self.phi_train = None

        # Model degree, proportional to the learnable parameters
        self.model_degree = model_degree

        # The least squares empirical risk minimization solution
        self.theta_erm = None

        # Generate the training data.
        self.x = np.array(x_train)
        self.y = np.array(y_train)

    def create_train_features(self):
        """
        Convert data points to feature matrix: phi=[x0^0,x0^1,x0^2...;x1^0,x1^1,x1^2...;x2^0,x2^1,x2^2...]
        :return: phi: training set feature matrix.
        """

        # Create Feature matrix
        logger.info('Create train: num of features {}'.format(self.model_degree))
        self.phi_train = self.convert_to_features()
        logger.info('self.phi_train.shape: {}'.format(self.phi_train.shape))
        return self.phi_train

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

    @abstractmethod
    def convert_to_features(self, model_degree: int) -> np.ndarray:
        """ To override
        convert self.x to features
        phi = self.x
        """
        pass

    @staticmethod
    @abstractmethod
    def convert_point_to_features(x: float, model_degree: int) -> np.ndarray:
        """ To override """
        pass


class DataPolynomial(DataBase):

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


class DataFourier(DataBase):

    def convert_to_features(self) -> np.ndarray:
        """
        Convert the training point to feature matrix.
        :param model_degree: the assumed polynomial degree of the data.
        :return: phy = [x0^0 , x0^1, ... , x0^pol_degree; x1^0 , x1^1, ... , x1^pol_degree].T
        """
        phi = []
        n = 1

        for i in range(self.model_degree + 1):
            if i == 0:
                phi.append(np.array([1] * len(self.x)))
            elif i % 2 != 0:
                phi.append(np.sqrt(2) * np.cos(np.pi * n * self.x))
            else:
                phi.append(np.sqrt(2) * np.sin(np.pi * n * self.x))
                n += 1
        phi = np.asarray(phi)
        return phi

    @staticmethod
    def convert_point_to_features(x: np.ndarray, model_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param model_degree: number of learnable parameters.
        :return: phi = [x^0,x^1,x^2,...].T
        """
        phi = []
        n = 1
        for i in range(model_degree + 1):
            if i == 0:
                phi.append(1)
            elif i % 2 != 0:
                phi.append(np.sqrt(2) * np.cos(np.pi * n * x))
            else:
                phi.append(np.sqrt(2) * np.sin(np.pi * n * x))
                n += 1

        phi = np.asarray(phi)
        if len(phi.shape) == 1:
            phi = phi[:, np.newaxis]

        return phi


data_type_dict = {
    'polynomial': DataPolynomial,
    'fourier': DataFourier
}
