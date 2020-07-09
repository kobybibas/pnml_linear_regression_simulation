import logging
import os
import os.path as osp
import time

import hydra
import numpy as np
import pandas as pd
import psutil
import ray

from real_data_utils import download_regression_datasets, get_data, execute_trails

logger = logging.getLogger(__name__)


def execute_dataset_experiment(dataset_name: str, cfg) -> pd.DataFrame:
    # Get dataset statistic
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    x_all, y_all = get_data(dataset_name, cfg.data_dir, is_add_bias_term=cfg.is_add_bias_term)
    n_train, n_features = int(train_ratio * x_all.shape[0]), x_all.shape[1]

    # Define train set to evaluate
    trainset_sizes = np.arange(1, n_features, cfg.step_size_over_param).astype(int)
    if cfg.is_underparam_region is True:
        trainset_sizes_1 = np.linspace(n_features, n_train, cfg.num_points_under_params).round().astype(int)
        trainset_sizes = np.unique(np.append(trainset_sizes, trainset_sizes_1))
    trainset_sizes = trainset_sizes[::-1]  # Largest trainset is the slowest, so start with it
    logger.info('{}: [n_data, n_features]=[{} {}]'.format(dataset_name, x_all.shape[0], n_features))
    logger.info(trainset_sizes)

    # Submit tasks
    task_list = [execute_trails.remote(x_all, y_all, cfg.n_trails, trainset_size,
                                       dataset_name, i, len(trainset_sizes),
                                       is_eval_mdl=cfg.is_eval_mdl,
                                       is_adaptive_var=cfg.is_adaptive_var,
                                       is_standardize_feature=cfg.is_standardize_feature,
                                       is_standardize_samples=cfg.is_standardize_samples
                                       ) for i, trainset_size in enumerate(trainset_sizes)]
    return task_list


@hydra.main(config_name='../configs/real_data.yaml')
def execute_real_datasets_experiment(cfg):
    t0 = time.time()
    logger.info(cfg.pretty())
    out_path = os.getcwd()

    # Move from output directory to src
    os.chdir('../../src')
    logger.info('[cwd out_dir]=[{} {}]'.format(os.getcwd(), out_path))

    # Download data
    if cfg.is_get_all_datasets_statistics is True:
        t1 = time.time()
        datasets_df = download_regression_datasets(cfg.data_dir)
        datasets_df.to_csv(osp.join(out_path, 'datasets.csv'))
        logger.info('Finish download_regression_datasets in {:.2f} sec'.format(time.time() - t1))

    # Initialize multiprocess. Every trainset size in a separate process
    cpu_count = psutil.cpu_count()
    cpu_to_use = cfg.num_cpus if cfg.num_cpus > 0 else cpu_count
    logger.info('CPU count: {}. Executing on {}'.format(cpu_count, cpu_to_use))
    ray.init(local_mode=not cfg.is_multi_process, num_cpus=cpu_to_use)

    # Choose iterate on datasets
    ray_task_list, len_datasets = [], len(cfg.dataset_names)
    for i, dataset_name in enumerate(cfg.dataset_names):
        t1 = time.time()
        ray_task_list += execute_dataset_experiment(dataset_name, cfg)
        logger.info('[{}/{}] Finish submit dataset {} in {:.2f} sec'.format(i, len_datasets - 1, dataset_name,
                                                                            time.time() - t1))

    # Wait task experiment to finish
    res_list = []
    total_jobs = len(ray_task_list)
    for job_num in range(total_jobs):
        t1 = time.time()
        ready_id, ray_task_list = ray.wait(ray_task_list)
        res_i = ray.get(ready_id[0])
        res_list.append(res_i)
        logger.info('[{:04d}/{}] {}. Size [train val test]=[{:03d} {} {}] in {:3.1f} sec. Local time {:3.1f} sec'.
                    format(job_num, total_jobs - 1, res_i['dataset_name'],
                           res_i['trainset_size'], res_i['valset_size'], res_i['testset_size'],
                           time.time() - t1, res_i['time']))

    res_df = pd.DataFrame(res_list)
    res_df = res_df.sort_values(by=['dataset_name', 'task_index'])
    out_file = osp.join(out_path, 'results.csv')
    res_df.to_csv(out_file, index=False)
    logger.info('Finish! in {:.2f} sec. {}'.format(time.time() - t0, out_file))


if __name__ == "__main__":
    execute_real_datasets_experiment()
