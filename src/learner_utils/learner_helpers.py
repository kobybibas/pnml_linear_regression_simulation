import logging

import numpy as np
import numpy.linalg as npl
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def fit_least_squares_estimator(x_arr: np.ndarray, y_vec: np.ndarray, lamb: float = 0.0) -> np.ndarray:
    """
    Fit least squares estimator
    :param x_arr: The training set features matrix. Each row represents an example.
    :param y_vec: the labels vector.
    :param lamb: regularization term.
    :return: the fitted parameters. A column vector
    """
    n, m = x_arr.shape
    phi_t_phi_plus_lamb = x_arr.T @ x_arr + lamb * np.eye(m)

    # If invertible, regular least squares
    if npl.cond(phi_t_phi_plus_lamb) < 1 / np.finfo('float').eps:
        inv = npl.inv(phi_t_phi_plus_lamb)
        theta = inv @ x_arr.T @ y_vec
    else:  # minimum norm
        # inv = npl.pinv(phi_arr @ phi_arr.T)
        # theta = phi_arr.T @ inv @ y
        reg = LinearRegression().fit(x_arr, y_vec)  # using scipy is more stable
        theta = reg.coef_

    theta = np.expand_dims(theta, 1)
    return theta


def calc_theta_norm(theta: np.ndarray) -> float:
    return npl.norm(theta) ** 2


def calc_square_error(x_data: np.ndarray, y_labels: np.ndarray, theta: np.ndarray) -> np.ndarray:
    y_hat = x_data @ theta
    return (y_hat - y_labels) ** 2


def calc_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta: np.ndarray, var: float) -> np.ndarray:
    y_hat = x_arr @ theta
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss


def calc_var_with_valset(x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray) -> float:
    y_hat = x_val @ theta
    return float(np.var(y_hat - y_val))


def calc_best_var(phi_arr: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    y_hat = phi_arr @ theta
    epsilon_square = (y_hat.squeeze() - y.squeeze()) ** 2
    var = np.mean(epsilon_square)
    return float(var)


def calc_effective_trainset_size(h_square: np.ndarray, rank: int, eigenvalue_threshold: float = 1e-12) -> int:
    """
    Calculate the effective dimension of the training set.
    :param h_square: the singular values of the trainset correlation matrix
    :param rank: the training set size
    :param eigenvalue_threshold: the smallest eigenvalue that is allowed
    :return: The effective dim
    """
    rank_effective = min(np.sum(h_square > eigenvalue_threshold), rank)
    return int(rank_effective)
