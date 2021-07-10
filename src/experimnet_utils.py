import copy
import logging
import time

import numpy as np
import pandas as pd
import ray

from data_utils.synthetic_data_utils import DataBase
from learner_utils.learner_helpers import calc_theta_norm
from learner_utils.pnml_utils import choose_pnml_h_type

logger = logging.getLogger(__name__)


def execute_x_vec(x_test: np.ndarray, data_h: DataBase, pnml_handlers: dict, x_bot_threshold: float) -> pd.DataFrame:
    """
    For each x point, calculate its regret.
    :param x_test: x array that contains the x points that will be calculated
    :param data_h: data handler class, used for training set.
    :param pnml_handlers: The learner class handlers in a dict.
    :param x_bot_threshold: Min valid value of |x_\bot|^2/|x|^2 to not considered as 0.
    :return: list of the calculated regret.
    """

    # Iterate on test samples
    t0 = time.time()
    ray_task_list = []
    for i, x_test_i in enumerate(x_test):
        # Eval pNML
        ray_task = execute_x_test.remote(
            x_test_i, data_h, copy.deepcopy(pnml_handlers), x_bot_threshold)
        ray_task_list.append(ray_task)
    logger.info(
        'Finish submitting tasks in {:.2f} sec'.format(time.time() - t0))

    # collect results
    res_list = []
    total_jobs = len(ray_task_list)
    logger.info('Collecting jobs. total_jobs={}'.format(total_jobs))
    for job_num in range(total_jobs):
        t1 = time.time()
        ready_id, ray_task_list = ray.wait(ray_task_list)
        res_i = ray.get(ready_id[0])
        res_list.append(res_i)

        # Report
        x_test = res_i['x_test']
        logger.info('[{:04d}/{}] Finish x={}. in {:3.1f}s.'.format(job_num,
                    total_jobs - 1, x_test, time.time() - t1))

    # Save to file
    res_df = pd.DataFrame(res_list)
    res_df = res_df.sort_values(by=['x_test'], ascending=[True])
    return res_df


@ray.remote
def execute_x_test(x_test_i: float, data_h: DataBase, pnml_handlers: dict, x_bot_threshold: float) -> dict:
    x_i = data_h.convert_point_to_features(x_test_i, data_h.model_degree)

    # Check which pnml to use
    pnml_h, x_bot_square, x_norm_square = choose_pnml_h_type(
        pnml_handlers, x_i, x_bot_threshold)

    # Calc normalization factor (nf)
    pnml_h.reset()
    nf = pnml_h.calc_norm_factor(x_i)
    regret = np.log(nf)
    success, msg = pnml_h.verify_pnml_results()

    # Analytical pnml
    analytical_nf = pnml_h.calc_analytical_norm_factor(x_i)
    analytical_regret = np.log(analytical_nf)

    trainset_size, num_features = pnml_h.x_arr_train.shape
    x_i = pnml_h.convert_to_column_vec(x_i)
    y_hat_erm = float(pnml_h.theta_erm.T @ x_i)

    theta_erm_norm = calc_theta_norm(pnml_h.theta_erm)
    return {'x_test': x_test_i, 'nf': nf, 'regret': regret, 'y_hat_erm': y_hat_erm,
            'analytical_nf': analytical_nf, 'analytical_regret': analytical_regret,
            'x_bot_square': x_bot_square, 'x_square': x_norm_square, 'success': success,
            'trainset_size': trainset_size, 'num_features': num_features,
            'theta_erm_norm': theta_erm_norm}
