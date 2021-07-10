import os.path as osp
from glob import glob

import numpy as np
import pandas as pd
import yaml


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


def plot_confidence_interval(ax, x, mean, std, count, color: str, alpha=0.3):
    mean, std, count = np.array(mean), np.array(std), np.array(count)
    lower = mean - 1.960 * std / np.sqrt(count)
    upper = mean + 1.960 * std / np.sqrt(count)
    ax.fill_between(x, upper, lower, color=color, alpha=alpha)  # std curves


def plot_logloss(ax, res_dict: dict, colors: list, alpha: float = 0.6,
                 label1="Minimum norm", label2="pNML", is_plot_first=True, alpha_conf=0.3):
    mean_df = res_dict["mean_df"]
    std_df = res_dict["std_df"]
    count_df = res_dict["count_df"]
    trainset_size = mean_df["trainset_size"].values
    num_features = mean_df['num_features']
    m_over_n = num_features / trainset_size

    if is_plot_first is True:
        key = "mn_test_logloss"
        ax.plot(m_over_n, mean_df[key], label=label1,
                color=colors[0], alpha=alpha)
        plot_confidence_interval(
            ax, m_over_n, mean_df[key], std_df[key], count_df[key], colors[0], alpha=alpha_conf
        )

    key = "pnml_test_logloss"
    ax.plot(m_over_n, mean_df[key], label=label2, color=colors[1], alpha=alpha)
    plot_confidence_interval(
        ax, m_over_n, mean_df[key], std_df[key], count_df[key], colors[1], alpha=alpha_conf
    )
    return ax


def plot_regret(ax, res_dict: dict, colors: list, alpha: float = 0.6):
    mean_df = res_dict["mean_df"]
    std_df = res_dict["std_df"]
    count_df = res_dict["count_df"]
    trainset_size = mean_df["trainset_size"].values
    num_features = mean_df['num_features']
    m_over_n = num_features / trainset_size

    key = "pnml_regret"
    ax.plot(m_over_n, mean_df[key], label="Empirical",
            color=colors[0], alpha=alpha)
    plot_confidence_interval(
        ax, m_over_n, mean_df[key], std_df[key], count_df[key], colors[0],
    )

    key = "analytical_pnml_regret"
    ax.plot(m_over_n, mean_df[key], label="Analytical",
            color=colors[1], alpha=alpha)
    plot_confidence_interval(
        ax, m_over_n, mean_df[key], std_df[key], count_df[key], colors[1])
    return ax


def calc_performance_per_regret(df: pd.DataFrame):
    splits = df.split.unique()
    count, regrets = np.histogram(df["pnml_regret"],
                                  bins=np.logspace(
                                      np.log10(
                                          df.analytical_pnml_regret.min()),
                                      np.log10(
                                          df.analytical_pnml_regret.max()),
                                      100))
    regrets = regrets[1:]

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
