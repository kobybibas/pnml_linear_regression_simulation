import math

import numpy as np


def getfloat(name):
    return float(name.split("_")[-1].split(".np")[0])


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def calc_nf_from_simulation(
        genie_file: str, sigma_square: float = 1e-3
) -> (float, np.ndarray):
    loaded_dict = np.load(genie_file).item()
    epsilons = loaded_dict["epsilons"]
    y_vec = loaded_dict["y_vec"]
    genies_probs = ((2 * np.pi * sigma_square) ** (-1 / 2)) * np.exp(
        -(epsilons ** 2) / (2 * sigma_square)
    ).squeeze()

    # genies_probs = medfilt(genies_probs.squeeze(), k=3)
    nf_simulation = np.trapz(genies_probs, x=y_vec)
    loaded_dict["genies_probs_list"] = genies_probs.squeeze()
    return nf_simulation, genies_probs, y_vec


def execute_x_test_regret(
        x_test: np.ndarray,
        model_degree: int,
        sigma_square,
        constant_dict: dict,
        genie_file: str,
        debug_dict=None,
):
    # Get constant
    P_N = constant_dict["P_N"]
    theta_mn_P_N_theta_mn = constant_dict["theta_mn_P_N_theta_mn"]
    u = constant_dict["u"]
    h_square = constant_dict["h_square"]
    theta_mn = constant_dict["theta_mn"]
    data_h = constant_dict["data_h"]
    phi_train = constant_dict["phi_train"]
    M, N = phi_train.shape

    # Convert to features
    phi_test = data_h.convert_point_to_features(x_test, model_degree)

    theta_T_P_N_x = ((theta_mn.T.dot(P_N).dot(phi_test)) ** 2) / theta_mn_P_N_theta_mn

    # x^T P_N^|| x
    x_P_N_x = np.sum(((u.T.dot(phi_test)).squeeze() ** 2 / h_square)[:N])
    regret_analytical = np.log(
        2 * math.gamma(5 / 4) / np.sqrt(np.pi)
        + math.gamma(7 / 4)
        / (6 * np.sqrt(np.pi))
        * (1 + x_P_N_x + 3.25 * theta_T_P_N_x)
    )

    # ||x_\bot||^2
    x_bot_square = np.sum(((u.T.dot(phi_test)) ** 2)[N:])

    # \sigma^2
    if sigma_square is None:
        sigma_square = (x_bot_square ** 2) * theta_mn_P_N_theta_mn

    # Simulation
    nf_simulation, genies_probs, y_vec = calc_nf_from_simulation(
        genie_file, sigma_square
    )
    regret_simulation = np.log(nf_simulation)

    # Add to debug dict
    if debug_dict is not None:
        debug_dict["x_bot_square"].append(x_bot_square)
        debug_dict["sigma_square"].append(sigma_square)
        debug_dict["x_P_N_x"].append(x_P_N_x)
        debug_dict["genies_probs"].append(genies_probs)
        debug_dict["y_vec"].append(y_vec)
    if nf_simulation < 1:
        print("Warning: ", x_test, nf_simulation)
    return regret_simulation, regret_analytical, debug_dict
