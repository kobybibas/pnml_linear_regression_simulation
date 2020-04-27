import copy
import logging
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from data_utils import DataBase
from pnml_utils import Pnml

logger = logging.getLogger(__name__)


def execute_x_vec(x_test_array, data_h: DataBase, pnml_h, is_mp: bool, out_dir: str) -> list:
    # Initialize output
    save_dir_genies_outputs = osp.join(out_dir, 'genies_output')
    save_file_name = osp.join(out_dir, 'res_model_degree_{}_lamb_{}.npy'.format(data_h.model_degree, pnml_h.lamb))
    save_theta_erm_file_name = osp.join(out_dir,
                                        'res_theta_erm_model_degree_{}_lamb_{}.npy'.format(data_h.model_degree,
                                                                                           pnml_h.lamb))
    np.save(save_theta_erm_file_name, pnml_h.theta_erm)
    os.makedirs(save_dir_genies_outputs, exist_ok=True)

    # iterate on test samples
    if is_mp is False:
        res_list = execute_x_test_array(x_test_array, data_h, pnml_h, save_file_name, save_dir_genies_outputs)
    else:
        res_list = execute_x_test_array_mp(x_test_array, data_h, pnml_h, save_file_name, save_dir_genies_outputs)
    logger.info('Save to {}'.format(save_file_name))
    np.save(save_file_name, res_list)

    return res_list


def execute_x_test(x_test: float, data_h: DataBase, pnml_h: Pnml, save_dir_genies_outputs: str) -> dict:
    phi_test = data_h.convert_point_to_features(x_test, data_h.model_degree)
    y_hat_erm = pnml_h.predict_erm(phi_test)
    regret = pnml_h.execute_regret_calc(phi_test)

    # Save genies products
    np.save(osp.join(save_dir_genies_outputs, f'genies_outputs_{x_test}.npy'), pnml_h.genies_output)
    return {'x_test': x_test, 'regret': regret, 'y_hat_erm': y_hat_erm}


def execute_x_test_array(x_test_array: np.ndarray, data_h: DataBase, pnml_h: Pnml,
                         save_file_name: str, save_dir_genies_outputs: str) -> list:
    res_list = []
    for i, x_test in enumerate(x_test_array):
        res = execute_x_test(x_test, data_h, pnml_h, save_dir_genies_outputs)
        res_list.append(res)
        logger.info('[{}/{}] x_test={}. Save to {}'.format(i, len(x_test_array), x_test, save_file_name))
        np.save(save_file_name, res_list)
    return res_list


def execute_x_test_array_mp(x_test_array: np.ndarray, data_h: DataBase, pnml_h: Pnml,
                            save_file_name: str, save_dir_genies_outputs: str) -> list:
    pool = mp.Pool()
    logger.info('mp.cpu_count: {}'.format(mp.cpu_count()))
    results = []
    pbar = tqdm(total=len(x_test_array))

    def log_result(result):
        results.append(result)
        pbar.update()

    for i, x_test in enumerate(x_test_array):
        pool.apply_async(execute_x_test,
                         args=(x_test, copy.deepcopy(data_h), copy.deepcopy(pnml_h), save_dir_genies_outputs),
                         callback=log_result)

    pool.close()
    pool.join()
    pbar.close()
    res_list = sorted(results, key=lambda k: k['x_test'])
    np.save(save_file_name, res_list)
    return res_list
