import logging
import os
import os.path as osp
import time

import hydra
import numpy as np
import pandas as pd
import psutil
import ray

from data_utils.real_data_utils import create_trainset_sizes_to_eval, execute_trail
from data_utils.real_data_utils import download_regression_datasets, get_pmlb_data, random_split_dataset

logger = logging.getLogger(__name__)


def get_uci_data(dataset_name: str, data_dir: str, train_test_split_num: int):
    data_txt_path = osp.join(data_dir, dataset_name, 'data', 'data.txt')
    data = np.loadtxt(data_txt_path)

    index_features_path = osp.join(data_dir, dataset_name, 'data', 'index_features.txt')
    index_features = np.loadtxt(index_features_path)

    index_target_path = osp.join(data_dir, dataset_name, 'data', 'index_target.txt')
    index_target = np.loadtxt(index_target_path)

    x_all = data[:, [int(i) for i in index_features.tolist()]]
    y_all = data[:, int(index_target.tolist())]

    # Add bias term
    x_all = np.hstack((x_all, np.ones((x_all.shape[0], 1))))

    # Load split file
    index_train_path = osp.join(data_dir, dataset_name, 'data', f'index_train_{train_test_split_num}.txt')
    index_train = np.loadtxt(index_train_path).astype(int)
    index_test_path = osp.join(data_dir, dataset_name, 'data', f'index_test_{train_test_split_num}.txt')
    index_test = np.loadtxt(index_test_path).astype(int)

    # Train-test split
    x_train, y_train = x_all[index_train], y_all[index_train]
    x_test, y_test = x_all[index_test], y_all[index_test]

    # Train-val split
    num_training_examples = int(0.9 * x_train.shape[0])
    x_val, y_val = x_train[num_training_examples:, :], y_train[num_training_examples:]
    x_train, y_train = x_train[:num_training_examples, :], y_train[:num_training_examples]

    return x_train, y_train, x_val, y_val, x_test, y_test


def submit_dataset_experiment_jobs(dataset_name: str, cfg) -> pd.DataFrame:
    if 'train_test_split_num' in cfg:
        iterator = cfg.train_test_split_num
    else:
        iterator = range(cfg.n_trails)

    task_list = []
    for trail_num in iterator:
        # Get dataset statistic
        if 'UCI_Datasets' in dataset_name:
            x_train, y_train, x_val, y_val, x_test, y_test = get_uci_data(dataset_name, cfg.data_dir,
                                                                          cfg.train_test_split_num[trail_num])
        else:
            x_all, y_all = get_pmlb_data(dataset_name, cfg.data_dir)
            # Randomly split dataset
            x_train, y_train, x_val, y_val, x_test, y_test = random_split_dataset(x_all, y_all,
                                                                                  is_standardize_feature=cfg.is_standardize_feature,
                                                                                  is_standardize_samples=cfg.is_standardize_samples,
                                                                                  is_add_bias_term=cfg.is_add_bias_term)

        # Define train set to evaluate
        n_train, n_features = x_train.shape
        trainset_sizes = create_trainset_sizes_to_eval(cfg.trainset_sizes, n_train, n_features, cfg.num_trainset_sizes)
        logger.info('{} trail_num={}: [n_features n_train n_val n_test]=[{} {} {} {}]'.format(
            dataset_name, trail_num, n_features, n_train, len(x_val), len(x_test)))
        logger.info(trainset_sizes)

        # Submit tasks
        for i, trainset_size in enumerate(trainset_sizes):

            # Reduce training set size
            x_train_reduced, y_train_reduced = x_train[:trainset_size, :], y_train[:trainset_size]
            if cfg.fast_dev_run is True:
                x_train, y_train = x_train[:3, :], y_train[:3]
                x_val, y_val = x_val[:2, :], y_val[:2]
                x_test, y_test = x_test[:2, :], y_test[:2]

            # Execute trail
            ray_task = execute_trail.remote(np.copy(x_train_reduced), np.copy(y_train_reduced),
                                            np.copy(x_val), np.copy(y_val),
                                            np.copy(x_test), np.copy(y_test),
                                            trail_num, trainset_size, dataset_name,
                                            is_eval_mdl=cfg.is_eval_mdl,
                                            is_eval_empirical_pnml=cfg.is_eval_empirical_pnml,
                                            is_eval_analytical_pnml=cfg.is_eval_analytical_pnml)
            task_list.append(ray_task)
    return task_list


def collect_dataset_experiment_results(ray_task_list: list):
    """
    Wait task experiment to finish
    :param ray_task_list: list of jobs
    :return: dataframe with results
    """

    res_list = []
    total_jobs = len(ray_task_list)
    logger.info('Collecting jobs. total_jobs={}'.format(total_jobs))
    for job_num in range(total_jobs):
        t1 = time.time()
        ready_id, ray_task_list = ray.wait(ray_task_list)
        res_i = ray.get(ready_id[0])
        res_list.append(res_i)

        # Report
        dataset_name = res_i['dataset_name'][0]
        n_trainset, n_valset, n_testset = res_i['trainset_size'][0], res_i['valset_size'][0], res_i['testset_size'][0]
        logger.info('[{:04d}/{}] {}. Size [train val test]=[{:03d} {} {}] in {:3.1f}s.'.format(
            job_num, total_jobs - 1, dataset_name, n_trainset, n_valset, n_testset, time.time() - t1))

    # Save to file
    res_df = pd.concat(res_list, ignore_index=True, sort=False)
    res_df = res_df.sort_values(by=['dataset_name', 'num_features', 'trainset_size', 'trail_num'],
                                ascending=[False, True, True, True])
    return res_df


def ray_init(cpu_num: int, is_local_mode: bool):
    """
    Initialize multiprocess. Every trainset size in a separate process
    :param cpu_num: NUmber of cpu to use. -1 to use all available
    :param is_local_mode: whether to run in the local computer. Used for debug.
    :return:
    """
    cpu_count = psutil.cpu_count()
    cpu_to_use = cpu_num if cpu_num > 0 else cpu_count
    cpu_to_use = cpu_to_use if is_local_mode is False else 1
    logger.info('cpu_count={}. Executing on {}. is_local_mode={}'.format(cpu_count, cpu_to_use, is_local_mode))
    ray.init(local_mode=is_local_mode, num_cpus=cpu_to_use)


@hydra.main(config_name='../configs/real_data.yaml')
def execute_real_datasets_experiment(cfg):
    t0 = time.time()
    logger.info(cfg.pretty())
    out_path = os.getcwd()

    # Set seed for debugging
    if cfg.seed >= 0:
        np.random.seed(cfg.seed)

    # Move from output directory to src
    os.chdir('../../src')
    logger.info('[cwd out_dir]=[{} {}]'.format(os.getcwd(), out_path))

    # Download data
    if cfg.is_datasets_statistics is True:
        download_regression_datasets(cfg.data_dir, out_path)

    # Initialize multiprocess. Every trainset size in a separate process
    ray_init(cfg.num_cpus, cfg.is_local_mode)

    # Iterate on datasets: send jobs
    ray_task_list, n_datasets = [], len(cfg.dataset_names)
    for i, dataset_name in enumerate(cfg.dataset_names):
        t1 = time.time()
        ray_task_list += submit_dataset_experiment_jobs(dataset_name, cfg)
        logger.info('[{}/{}] Submitted {} in {:.2f} sec'.format(i, n_datasets - 1, dataset_name, time.time() - t1))

    # Collect results
    res_df = collect_dataset_experiment_results(ray_task_list)
    out_file = osp.join(out_path, 'results.csv')
    res_df.to_csv(out_file, index=False)
    logger.info('Finish! in {:.2f} sec. {}'.format(time.time() - t0, out_file))


if __name__ == "__main__":
    execute_real_datasets_experiment()
