import logging
from abc import abstractmethod

import numpy as np
import numpy.linalg as npl

logger = logging.getLogger(__name__)


class DataBase:
    def __init__(self, x_train: list, y_train: list, model_degree: int):
        # Model degree, proportional to the learn-able parameters
        self.model_degree = model_degree

        # The least squares empirical risk minimization solution
        self.theta_erm = None

        # Generate the training data.
        self.x = np.array(x_train)
        self.y = np.array(y_train)

        # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x)
        self.phi_train = self.create_train_features()
        logger.info('self.phi_train.shape: {}'.format(self.phi_train.shape))
        trainset_size, _ = self.phi_train.shape
        u, h, v_t = npl.svd(self.phi_train)
        assert trainset_size == len(self.y)
        logger.info('Training set eigenvalues: {}'.format(h))

    def create_train_features(self) -> np.ndarray:
        """
        Convert data points to feature matrix: phi=[x0^0,x0^1,x0^2...;x1^0,x1^1,x1^2...;x2^0,x2^1,x2^2...]
        Each row corresponds to feature vector.
        :return: phi: training set feature matrix.
        """
        phi_train = []
        for x_i in self.x:
            phi_train_i = self.convert_point_to_features(x_i, self.model_degree)

            # Convert column vector to raw and append
            phi_train.append(np.squeeze(phi_train_i.T))
        phi_train = np.asarray(phi_train)
        return phi_train

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
    @abstractmethod
    def convert_point_to_features(x: float, model_degree: int) -> np.ndarray:
        """ To override """
        pass


class DataPolynomial(DataBase):

    @staticmethod
    def convert_point_to_features(x: float, pol_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param pol_degree: the assumed polynomial degree.
        :return: phi = [x^0,x^1,x^2,...], row vector
        """
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(x, n))
        phi = np.asarray(phi)
        phi = np.expand_dims(phi, 1)
        phi = phi / npl.norm(phi)
        return phi


class DataFourier(DataBase):

    @staticmethod
    def convert_point_to_features(x: np.ndarray, model_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param model_degree: number of learn-able parameters.
        :return: phi = [x^0,x^1,x^2,...], row vector
        """
        phi = []
        n = 1
        for i in range(model_degree):
            if i == 0:
                phi_i = 1
            elif i % 2 != 0:
                phi_i = np.sqrt(2) * np.cos(np.pi * n * x)
            else:
                phi_i = np.sqrt(2) * np.sin(np.pi * n * x)
                n += 1
            phi.append(phi_i)
        phi = np.asarray(phi)
        phi = np.expand_dims(phi, 1)

        phi = phi/npl.norm(phi)
        return phi


data_type_dict = {
    'polynomial': DataPolynomial,
    'fourier': DataFourier
}
