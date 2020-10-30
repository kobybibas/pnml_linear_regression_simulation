import os.path as osp
from glob import glob

import pandas as pd
import yaml


def getfloat(name):
    return float(name.split("_")[-1].split(".np")[0])


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
