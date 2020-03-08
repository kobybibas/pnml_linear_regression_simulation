import multiprocessing as mp
import os.path as osp

import numpy as np
from loguru import logger
# import os.path as osp
#
# import numpy as np
# import pandas as pd
# from loguru import logger
from tqdm import tqdm

from data_utils import DataPolynomial, DataCosine


#
# from data_utils import DataParameters
# from data_utilities import DataPolynomial
# from pnml_joint_utilities import PnmlJoint
# from pnml_min_norm_utils import PnmlMinNorm
# from pnml_utilities import PNMLParameters
# from pnml_utilities import Pnml


def execute_x_vec(pnml_h, poly_degree: int, lamb: float, params: dict, out_dir: str):
    # Initialize output
    save_file_name = osp.join(out_dir, 'res_poly_deg_{}_lamb_{}.npy'.format(poly_degree, lamb))

    # Create trainset
    if params['data_type'] == 'poly':
        data_h = DataPolynomial(params['x_train'], params['y_train'])
    elif params['data_type'] == 'cosine':
        data_h = DataCosine(params['x_train'], params['y_train'])
    else:
        raise ValueError('params.data_type={} no recognized'.format(params['data_type']))
    phi_train = data_h.create_train(poly_degree)
    y_train = data_h.y

    theta_erm = data_h.fit_least_squares_estimator(phi_train, y_train, lamb)
    pnml_h.set_erm_predictor(theta_erm)
    pnml_h.set_y_interval(params['y_min'], params['y_max'], params['dy'])
    pnml_h.min_sigma_square = params['min_sigma_square']

    # iterate on test samples
    x_test_array = np.arange(params['x_test_min'], params['x_test_max'], params['dx_test']).round(2)
    if params['multi_process'] is False:
        res_list = execute_x_test_array(x_test_array, data_h, pnml_h, theta_erm, poly_degree,
                                        phi_train, y_train, lamb,
                                        save_file_name)
    else:
        res_list = execute_x_test_array_mp(x_test_array, data_h, pnml_h,
                                           theta_erm, phi_train, y_train, poly_degree, lamb)
    logger.info('Save to {}'.format(save_file_name))
    np.save(save_file_name, res_list)
    return res_list


def execute_x_test(x_test: float,
                   data_h: DataPolynomial, pnml_h,
                   theta_erm: np.ndarray, phi_train: np.ndarray, y_train: np.ndarray, poly_degree: int, lamb: float):
    phi_test = data_h.convert_point_to_features(x_test, poly_degree)
    y_hat_erm = pnml_h.predict_erm(theta_erm, phi_test)

    regret = pnml_h.execute_regret_calc(phi_test, phi_train, y_train, lamb)
    return {'x_test': x_test, 'regret': regret, 'y_hat_erm': y_hat_erm}


def execute_x_test_array(x_test_array: np.ndarray, data_h: DataPolynomial, pnml_h,
                         theta_erm: np.ndarray, poly_degree: int,
                         phi_train: np.ndarray, y_train: np.ndarray, lamb: float,
                         save_file_name):
    res_list = []
    for i, x_test in enumerate(x_test_array):
        res = execute_x_test(x_test, data_h, pnml_h,
                             theta_erm, phi_train, y_train,
                             poly_degree, lamb)
        res_list.append(res)
        logger.info('[{}/{}] x_test={}. Save to {}'.format(i, len(x_test_array), x_test, save_file_name))
        np.save(save_file_name, res_list)
    return res_list


def execute_x_test_array_mp(x_test_array, data_h, pnml_h,
                            theta_erm, phi_train, y_train,
                            poly_degree, lamb):
    pool = mp.Pool()
    logger.info('mp.cpu_count: {}'.format(mp.cpu_count()))
    results = []
    pbar = tqdm(total=len(x_test_array))

    def log_result(result):
        results.append(result)
        pbar.update()

    for i, x_test in enumerate(x_test_array):
        pool.apply_async(execute_x_test,
                         args=(x_test, data_h, pnml_h,
                               theta_erm, phi_train, y_train,
                               poly_degree, lamb),
                         callback=log_result)

    pool.close()
    pool.join()
    pbar.close()
    results_sorted = sorted(results, key=lambda k: k['x_test'])
    return results_sorted
