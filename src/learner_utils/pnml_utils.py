import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as npl

from learner_utils.learner_helpers import fit_least_squares_estimator

logger_default = logging.getLogger(__name__)




def choose_pnml_h_type(pnml_handlers, x_i, x_bot_threshold):
    overparam_pnml_h = pnml_handlers['overparam']
    underparam_pnml_h = pnml_handlers['underparam']

    x_norm_square = np.linalg.norm(x_i) ** 2
    if overparam_pnml_h.rank >= overparam_pnml_h.m:
        pnml_h = underparam_pnml_h
        x_bot_square = 0.0
    else:
        x_bot_square = overparam_pnml_h.calc_x_bot_square(x_i)
        pnml_h = overparam_pnml_h if x_bot_square / x_norm_square > x_bot_threshold else underparam_pnml_h
    return pnml_h, x_bot_square, x_norm_square


class BasePNML:
    def __init__(self, x_arr_train: np.ndarray, y_vec_train: np.ndarray, lamb: float = 0.0, var: float = 1e-1,
                 logger=logger_default):
        __metaclass__ = ABCMeta
        assert x_arr_train.shape[0] == y_vec_train.shape[0]

        # Train feature matrix and labels
        self.x_arr_train = x_arr_train
        self.y_vec_train = y_vec_train

        # Regularization term
        self.lamb = lamb

        # Default variance
        self.var_input = var
        self.var = var

        # ERM least squares parameters
        self.n, self.m = self.x_arr_train.shape
        self.rank = min(self.m, self.n)
        self.u, self.h, self.vt = npl.svd(self.x_arr_train.T)
        self.h_square = self.h ** 2
        self.theta_erm = fit_least_squares_estimator(self.x_arr_train, self.y_vec_train, lamb=self.lamb)
        self.rank = np.sum(self.h_square > np.finfo('float').eps)

        # Indication of success or fail
        self.intermediate_dict = {}
        self.logger = logger

    def reset(self):
        self.intermediate_dict = {}
        self.var = self.var_input

    @abstractmethod
    def optimize_var(self, x_test: np.ndarray, y_gt: float) -> (float, float):
        pass

    @abstractmethod
    def calc_norm_factor(self, x_test: np.array) -> float:
        pass

    def add_test_to_train(self, x_arr_train: np.ndarray, x_test: np.ndarray,
                          y_train_vec: np.ndarray, y_test: float) -> (np.ndarray, np.ndarray):
        """
        Add the test set feature to training set feature matrix
        :param x_arr_train: training set feature matrix.
        :param x_test: test set feature.
        :param y_train_vec: training labels.
        :param y_test: test label to add.
        :return: concat train and test.
        """
        x_test = self.convert_to_column_vec(x_test)

        # Concat train and test
        x_arr = np.concatenate((x_arr_train, x_test.T), axis=0)
        y_vec = np.append(y_train_vec.squeeze(), y_test)
        assert x_arr.shape[0] == y_vec.shape[0]
        return x_arr, y_vec

    @staticmethod
    def convert_to_column_vec(x):
        # Make the input as column vector
        x_col = np.copy(x)
        if len(x.shape) == 1:
            x_col = np.expand_dims(x, 1)

        # convert test to row vector if needed
        if x.shape[0] == 1:
            x_col = x.T
        return x_col

    @abstractmethod
    def fit_least_squares_estimator(self, x_arr: np.ndarray, y_vec: np.ndarray, lamb: float) -> np.ndarray:
        """
        Override this.
        :param x_arr: data matrix. Each row is a sample.
        :param y_vec: label vector.
        :param lamb: regularization term.
        :return:
        """
        pass

    @abstractmethod
    def calc_pnml_logloss(self, x_test: np.ndarray, y_gt: float, nf: float) -> float:
        pass

    @abstractmethod
    def calc_genie_logloss(self, x_test: np.ndarray, y_gt: float, nf: float) -> float:
        pass

    @abstractmethod
    def verify_pnml_results(self) -> (bool, str):
        success, msg = True, ''
        return success, msg
