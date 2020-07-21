import numpy as np


def compute_mse(x_data: np.ndarray, y_labels: np.ndarray, theta: np.ndarray) -> float:
    y_hat_labels = x_data @ theta
    return float(np.mean((y_hat_labels - y_labels) ** 2))


def compute_logloss(x_arr: np.ndarray, y: np.ndarray, theta: np.ndarray, var: float) -> float:
    y_hat = x_arr @ theta
    prob = np.exp(-(y_hat - y) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss.mean()


def estimate_sigma_with_valset(x_val: np.ndarray, y_val: np.ndarray, theta: np.ndarray) -> float:
    y_hat = x_val @ theta
    noise_std = np.std(y_hat - y_val)
    return noise_std


def estimate_sigma_unbiased(x_train: np.ndarray, y_train: np.ndarray, theta: np.ndarray, n_train: int,
                            num_features: int) -> float:
    y_hat = x_train @ theta
    noise_std = np.sum(np.square(y_train - y_hat)) / (n_train - num_features - 1)
    return noise_std


def estimate_base_var(x_train, y_train, x_val, y_val, theta_erm, is_adaptive_var: bool):
    # Compute variance
    n_train, num_features = x_train.shape
    if n_train > num_features:
        noise_std = estimate_sigma_unbiased(x_train, y_train, theta_erm, n_train, num_features)
    else:
        noise_std = estimate_sigma_with_valset(x_val, y_val, theta_erm) if is_adaptive_var is True else 1.0
    var = noise_std ** 2
    return var
