import os.path as osp
from glob import glob

import numpy as np
import yaml


def getfloat(name):
    return float(name.split("_")[-1].split(".np")[0])


def load_simulation_results(base_dir: str):
    # Load configuration file
    config_path = osp.join(base_dir, ".hydra", "config.yaml")
    with open(f"{config_path}") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    model_degree = params["model_degree"]
    x_train, y_train = params["x_train"], params["y_train"]

    # Load results file
    file_name = glob(osp.join(base_dir, f"res_model_degree_{model_degree}_lamb_*.npy"))[0]
    loaded_list = np.load(file_name, allow_pickle=True)
    loaded_list = sorted(loaded_list, key=lambda k: k["x_test"])

    # Get results
    x_test = [res["x_test"] for res in loaded_list]
    regret = [res["regret"] for res in loaded_list]
    y_hat_erm = [res["y_hat_erm"] for res in loaded_list]
    genies_output_path = osp.join(base_dir, 'genies_output')

    # Return res dict
    return {
        "model_degree": model_degree,
        "x_test": x_test,
        "regret": regret,
        "y_hat_erm": y_hat_erm,
        "x_train": x_train,
        "y_train": y_train,
        "base_dir": base_dir,
        "genies_output_path": genies_output_path,
        "params": params
    }


def load_genies_files(base_dir: str) -> list:
    genie_files = glob(osp.join(base_dir, "genies_outputs_*.npy"))
    genie_files.sort(key=lambda f: getfloat(f))
    return genie_files


def calc_nf_from_single_genie_file(genie_file: str, sigma_square: float = 1e-3) -> (float, np.ndarray):
    loaded_dict = np.load(genie_file, allow_pickle=True).item()
    epsilons, y_vec = loaded_dict["epsilons"], loaded_dict["y_vec"]
    genies_probs = (1 / np.sqrt(2 * np.pi * sigma_square)) * np.exp(-epsilons ** 2 / (2 * sigma_square))
    nf = np.trapz(genies_probs.squeeze(), x=y_vec)
    return nf, genies_probs, y_vec


def calc_nf_from_genie_files(genie_files_base_dir: str, sigma_square: float = 1e-3) -> (np.ndarray, list):
    # Load genies file
    genie_files = load_genies_files(genie_files_base_dir)

    nfs, debug_list = [], []
    for f in genie_files:
        # calculate normalization factor
        nf, probs, y_vec = calc_nf_from_single_genie_file(f, sigma_square)
        nfs.append(nf)
        debug_list.append({'nf': nf, 'probs': probs, 'y_vec': y_vec, 'file': f})
    regrets = np.log(nfs)
    return regrets, debug_list


def calc_theta_mn(x_arr: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    # x_arr: each row is feature vec

    n, m = x_arr.shape

    if n >= m:
        # under parameterized region
        inv = np.linalg.inv(x_arr.T @ x_arr)
        theta = inv @ x_arr.T @ y_vec
    else:
        # over parameterized region
        inv = np.linalg.inv(x_arr @ x_arr.T)
        theta = x_arr.T @ inv @ y_vec
    return theta
