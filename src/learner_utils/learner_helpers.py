import logging

import numpy as np
import numpy.linalg as npl

logger = logging.getLogger(__name__)


def fit_least_squares_estimator(phi_arr: np.ndarray, y: np.ndarray, lamb: float = 0.0) -> np.ndarray:
    """
    Fit least squares estimator
    :param phi_arr: The training set features matrix. Each row represents an example.
    :param y: the labels vector.
    :param lamb: regularization term.
    :return: the fitted parameters. A column vector
    """
    phi_t_phi = phi_arr.T @ phi_arr
    inv = npl.pinv(phi_t_phi + lamb * np.eye(phi_t_phi.shape[0], phi_t_phi.shape[1]))
    theta = inv @ phi_arr.T @ y
    theta = np.expand_dims(theta, 1)
    return theta


def calc_theta_norm(theta: np.ndarray) -> float:
    return npl.norm(theta) ** 2


def calc_mse(x_data: np.ndarray, y_labels: np.ndarray, theta: np.ndarray) -> np.ndarray:
    y_hat = x_data @ theta
    return (y_hat - y_labels) ** 2


def calc_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta: np.ndarray, var: float) -> np.ndarray:
    y_hat = x_arr @ theta
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss


def calc_var_with_valset(x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray) -> float:
    y_hat = x_val @ theta
    return np.var(y_hat - y_val)


def calc_best_var(phi_arr: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    y_hat = phi_arr @ theta
    epsilon_square = (y_hat.squeeze() - y.squeeze()) ** 2
    var = np.mean(epsilon_square)
    return float(var)
