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
        # inv = npl.pinv(x_arr @ x_arr.T)
        # theta = x_arr.T @ inv @ y_vec
        reg = LinearRegression(fit_intercept=False).fit(
            x_arr, y_vec)  # using scipy is more stable
        theta = reg.coef_

    theta = np.expand_dims(theta, 1)
    return theta


def calc_theta_norm(theta: np.ndarray) -> float:
    return npl.norm(theta) ** 2


def calc_square_error(x_data: np.ndarray, y_labels: np.ndarray, theta: np.ndarray) -> np.ndarray:
    y_hat = x_data @ theta
    y_hat = y_hat.squeeze()
    return np.squeeze((y_hat - y_labels) ** 2)


def calc_logloss(x_arr: np.ndarray, y_gt: np.ndarray, theta: np.ndarray, var: float) -> np.ndarray:
    y_hat = x_arr @ theta
    y_hat = y_hat.squeeze()

    logloss = 0.5 * np.log(2 * np.pi * var) + (y_gt - y_hat) ** 2 / (2 * var)
    return logloss.squeeze()


def calc_best_var(phi_arr: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    y_hat = phi_arr @ theta
    epsilon_square = (y_hat.squeeze() - y.squeeze()) ** 2
    var = np.mean(epsilon_square)
    return float(var)
