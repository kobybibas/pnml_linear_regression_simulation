import logging
import os
import os.path as osp
import time

import hydra
import numpy.linalg as npl
import pandas as pd
import ray

from data_utils.real_data_utils import choose_samples_for_debug
from data_utils.real_data_utils import create_trainset_sizes_to_eval, execute_trail, get_uci_data
from data_utils.real_data_utils import get_set_splits, execute_reduce_dataset
from ray_utils import ray_init

logger = logging.getLogger(__name__)


def submit_dataset_experiment_jobs(dataset_name: str, cfg) -> pd.DataFrame:
    logger_file_path = logging.getLoggerClass().root.handlers[1].baseFilename
    splits = cfg.splits if len(cfg.splits) > 0 else get_set_splits(
        dataset_name, cfg.data_dir)
    task_list = []
    for split in splits:
        trainset, valset, testset = get_uci_data(dataset_name, cfg.data_dir, split,
                                                 is_standardize_features=cfg.is_standardize_features,
                                                 is_add_bias_term=cfg.is_add_bias_term,
                                                 is_normalize_data=cfg.is_normalize_data
                                                 )
        # Define train set size to evaluate
        n_train, n_features = trainset[0].shape
        trainset_sizes = create_trainset_sizes_to_eval(
            cfg.trainset_sizes, n_train, n_features, cfg.max_train_samples)
        logger.info('{} split={}: [n_features n_train n_val n_test]=[{} {} {} {}]'.format(
            dataset_name, split, n_features, n_train, len(valset[0]), len(testset[0])))
        logger.info(trainset_sizes)

        # Submit tasks
        for i, trainset_size in enumerate(trainset_sizes):
            # Reduce training set size
            trainset_input, valset_input, testset_input = choose_samples_for_debug(
                cfg, trainset, valset, testset)
            x_train_reduced, y_train_reduced = execute_reduce_dataset(trainset_input[0], trainset_input[1],
                                                                      trainset_size)
            logger.info('{} split={} train.shape={}.\tTrainset svd [largest smallest]={}'.format(
                dataset_name, split, x_train_reduced.shape, npl.svd(x_train_reduced)[1][[0, -1]]))

            # Execute trail
            ray_task = execute_trail.remote(x_train_reduced, y_train_reduced,
                                            valset_input[0], valset_input[1], testset_input[0], testset_input[1],
                                            split, trainset_size, dataset_name,
                                            cfg.is_execute_empirical_pnml,
                                            pnml_optim_param=cfg.pnml_optim_param,
                                            debug_print=cfg.fast_dev_run or len(
                                                cfg.test_idxs) > 0,
                                            logger_file_path=logger_file_path)
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
        n_trainset, n_valset, n_testset = res_i['trainset_size'][
            0], res_i['valset_size'][0], res_i['testset_size'][0]
        logger.info('[{:04d}/{}] {}. Size [train val test]=[{:03d} {} {}] in {:3.1f}s.'.format(
            job_num, total_jobs - 1, dataset_name, n_trainset, n_valset, n_testset, time.time() - t1))

    # Save to file
    res_df = pd.concat(res_list, ignore_index=True, sort=False)
    res_df = res_df.sort_values(by=['dataset_name', 'num_features', 'trainset_size', 'split'],
                                ascending=[False, True, True, True])
    return res_df


@hydra.main(config_path='../configs', config_name="real_data")
def execute_real_datasets_experiment(cfg):
    t0 = time.time()
    logger.info(cfg.pretty())
    out_path = os.getcwd()

    # Move from output directory to src
    os.chdir(osp.join(hydra.utils.get_original_cwd(), 'src'))
    logger.info('[cwd out_dir]=[{} {}]'.format(os.getcwd(), out_path))

    # Initialize multiprocess. Every trainset size in a separate process
    ray_init(cfg.num_cpus, cfg.is_local_mode)

    # Iterate on datasets: send jobs
    t1 = time.time()
    ray_task_list = submit_dataset_experiment_jobs(cfg.dataset_name, cfg)
    logger.info('Submitted {} in {:.2f} sec'.format(
        cfg.dataset_name, time.time() - t1))

    # Collect results
    res_df = collect_dataset_experiment_results(ray_task_list)
    out_file = osp.join(out_path, 'results.csv')
    res_df.to_csv(out_file, index=False)
    logger.info('Finish! in {:.2f} sec. {}'.format(time.time() - t0, out_file))


if __name__ == "__main__":
    execute_real_datasets_experiment()
