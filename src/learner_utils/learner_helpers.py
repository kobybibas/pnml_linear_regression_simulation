import numpy as np
import numpy.linalg as npl


def calc_theta_norm(theta: np.ndarray) -> float:
    return npl.norm(theta)


def compute_mse(x_data: np.ndarray, y_labels: np.ndarray, theta: np.ndarray) -> np.ndarray:
    y_hat = x_data @ theta
    return (y_hat - y_labels) ** 2


def compute_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta: np.ndarray, var: float) -> np.ndarray:
    y_hat = x_arr @ theta
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss


def estimate_variance_with_valset(x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray) -> float:
    y_hat = x_val @ theta
    return np.var(y_hat - y_val)
