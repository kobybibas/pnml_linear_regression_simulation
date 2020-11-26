import logging
from abc import abstractmethod

import numpy as np
import numpy.linalg as npl

from learner_utils.learner_helpers import fit_least_squares_estimator

logger = logging.getLogger(__name__)


class DataBase:
    def __init__(self, x_train: list, y_train: list, model_degree: int):
        # Model degree, proportional to the learn-able parameters
        self.model_degree = model_degree

        # Generate the training data.
        self.x = np.array(x_train)
        self.y = np.array(y_train)

        # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x)
        self.phi_train = self.create_train_features()
        logger.info('self.phi_train.shape: {}'.format(self.phi_train.shape))
        trainset_size, _ = self.phi_train.shape
        self.u, self.h_square, self.v_t = npl.svd(self.phi_train.T @ self.phi_train)
        assert trainset_size == len(self.y)

        theta_erm = fit_least_squares_estimator(self.phi_train, self.y)
        mse = np.mean((self.y - np.squeeze(self.phi_train @ theta_erm)) ** 2)
        logger.info('Training set: mse={}, h_square: {}'.format(mse, self.h_square ** 2))
        self.pca_values_trainset = [self.execute_pca_dim_reduction(phi_i) for phi_i in self.phi_train]

    def execute_pca_dim_reduction(self, x_i):
        projections = self.u.T @ x_i
        return float(projections[0])

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

    @staticmethod
    @abstractmethod
    def convert_point_to_features(x: float, model_degree: int) -> np.ndarray:
        """ To override """
        pass

    def sweep_x_find_h_square(self):
        # Generate the training data.
        for i in range(10):
            self.x = np.random.uniform(low=0.0, high=1.0, size=5)

            # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x)
            self.phi_train = self.create_train_features()
            logger.info('self.phi_train.shape: {}'.format(self.phi_train.shape))
            u, h, v_t = npl.svd(self.phi_train)
            logger.info('sweep_x_find_h_square')
            logger.info('    x = {}'.format(self.x))
            logger.info('    h^2 = {}'.format(h ** 2))


class DataPolynomial(DataBase):

    @staticmethod
    def convert_point_to_features(x: float, pol_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param pol_degree: the assumed polynomial degree.
        :return: phi = [x^0,x^1,x^2,...], row vector
        """
        ns = np.arange(1, pol_degree, 1)
        phi = np.power(x, ns)
        # phi = phi / npl.norm(phi)
        phi = np.append(1, phi)
        phi = np.expand_dims(phi, 1)
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
        ns = np.arange(1, model_degree, 1)
        phi = np.cos(ns * np.pi * x + np.pi * ns / 2)
        # phi = phi / npl.norm(phi)
        phi = np.append(1, phi)
        phi = np.expand_dims(phi, 1)
        return phi


data_type_dict = {
    'polynomial': DataPolynomial,
    'fourier': DataFourier
}
