import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

cmap = plt.get_cmap("cubehelix")
indices = np.linspace(0, cmap.N, 10)
colors = [cmap(int(i)) for i in indices]



def load_simulation_results(base_dirs: list):
    dfs, cfgs = [], []
    for base_dir in base_dirs:
        csv_list = glob(osp.join(base_dir, "*.csv"))
        assert len(csv_list) == 1, base_dir
        csv_path = csv_list[0]
        df = pd.read_csv(csv_path)
        dfs.append(df)

        cfg_path = osp.join(base_dir, ".hydra", "config.yaml")
        with open(cfg_path, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        cfgs.append(cfg)

    df = pd.concat(dfs, ignore_index=True, sort=False)
    num_features_list = df.num_features.unique()
    return num_features_list, df


def plot_confidence_interval(ax, x, mean, std, count, color: str):
    mean, std, count = np.array(mean), np.array(std), np.array(count)
    lower = mean - 1.960 * std / np.sqrt(count)
    upper = mean + 1.960 * std / np.sqrt(count)
    ax.fill_between(x, upper, lower, color=color, alpha=0.2)  # std curves


def plot_logloss(ax, res_dict: dict):
    mean_df = res_dict["mean_df"]
    std_df = res_dict["std_df"]
    count_df = res_dict["count_df"]
    trainset_size = mean_df["trainset_size"].values

    key = "mn_test_logloss"
    ax.plot(trainset_size, mean_df[key], label="Minimum norm", color="C0", alpha=0.6)
    plot_confidence_interval(
        ax, trainset_size, mean_df[key], std_df[key], count_df[key], "C0"
    )

    key = "pnml_test_logloss"
    ax.plot(trainset_size, mean_df[key], label="pNML", color="C1", alpha=0.6)
    plot_confidence_interval(
        ax, trainset_size, mean_df[key], std_df[key], count_df[key], "C1"
    )
    return ax


def plot_regret(ax, res_dict: dict):
    mean_df = res_dict["mean_df"]
    std_df = res_dict["std_df"]
    count_df = res_dict["count_df"]
    trainsets = mean_df["trainset_size"].values

    key = "pnml_regret"
    ax.plot(trainsets, mean_df[key], label="Empirical", color=colors[3], alpha=0.6)
    plot_confidence_interval(
        ax, trainsets, mean_df[key], std_df[key], count_df[key], colors[3],
    )

    key = "analytical_pnml_regret"
    ax.plot(trainsets, mean_df[key], label="Analytical", color=colors[-4], alpha=0.6)
    plot_confidence_interval(ax, trainsets, mean_df[key], std_df[key], count_df[key], colors[-4], )
    return ax


def calc_performance_per_regret(df: pd.DataFrame, num_features: int):
    splits = df.split.unique()
    df = df[df.trainset_size == num_features]

    regrets = np.linspace(df.pnml_regret.min(), df.pnml_regret.max(), 100)

    pnml_losses, mn_losses, cdfs = [], [], []
    counts = np.zeros(len(regrets))
    for split in splits:
        df_split = df.loc[df.split == split]

        # For each regret threshold, calc performance
        pnml_loss_i, mn_loss_i, cdf_i = [], [], []
        for i, regret_thresh in enumerate(regrets):
            df_i = df_split[df_split.pnml_regret <= regret_thresh]
            pnml_loss_i.append(df_i["pnml_test_logloss"].mean())
            mn_loss_i.append(df_i["mn_test_logloss"].mean())
            cdf_i.append(df_i.shape[0] / df_split.shape[0])
            counts[i] += df_i.shape[0]
        pnml_losses.append(pnml_loss_i)
        mn_losses.append(mn_loss_i)
        cdfs.append(cdf_i)

    # Metrics
    pnml_losses_arr = np.asarray(pnml_losses)
    mn_losses_arr = np.asarray(mn_losses)
    cdfs_arr = np.asarray(cdfs)

    pnml_losses_arr = np.nan_to_num(pnml_losses_arr)
    mn_losses_arr = np.nan_to_num(mn_losses_arr)
    cdfs_arr = np.nan_to_num(cdfs_arr)

    pnml_means = pnml_losses_arr.mean(axis=0)
    mn_means = mn_losses_arr.mean(axis=0)
    cdf_means = cdfs_arr.mean(axis=0)

    pnml_stds = pnml_losses_arr.std(axis=0)
    mn_stds = mn_losses_arr.std(axis=0)
    cdf_stds = cdfs_arr.std(axis=0)
    return regrets, counts, pnml_means, mn_means, cdf_means, pnml_stds, mn_stds, cdf_stds
