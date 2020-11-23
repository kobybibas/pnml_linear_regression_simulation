import logging
import time

import numpy as np
import pandas as pd
import ray

from data_utils.synthetic_data_utils import DataBase
from learner_utils.analytical_pnml_utils import AnalyticalPNML
from learner_utils.pnml_utils import BasePNML

logger = logging.getLogger(__name__)


def execute_x_vec(x_test_array: np.ndarray, data_h: DataBase,
                  pnml_h: BasePNML, analytical_pnml_h: AnalyticalPNML) -> pd.DataFrame:
    """
    For each x point, calculate its regret.
    :param x_test_array: x array that contains the x points that will be calculated
    :param data_h: data handler class, used for training set.
    :param pnml_h: The learner class handler.
    :return: list of the calculated regret.
    """

    # Iterate on test samples
    t0 = time.time()
    ray_task_list = []
    for i, x_test in enumerate(x_test_array):
        ray_task = execute_x_test.remote(x_test, data_h, pnml_h, analytical_pnml_h)
        ray_task_list.append(ray_task)
    logger.info('Finish submitting tasks in {:.2f} sec'.format(time.time() - t0))

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
        logger.info('[{:04d}/{}] Finish x={}. in {:3.1f}s.'.format(job_num, total_jobs - 1, x_test, time.time() - t1))

    # Save to file
    res_df = pd.DataFrame(res_list)
    res_df = res_df.sort_values(by=['x_test'], ascending=[True])
    return res_df


@ray.remote
def execute_x_test(x_test: float, data_h: DataBase, pnml_h: BasePNML, analytical_pnml_h: AnalyticalPNML) -> dict:
    phi_test = data_h.convert_point_to_features(x_test, data_h.model_degree)
    y_hat_erm = pnml_h.predict_erm(phi_test)
    nf = pnml_h.calc_norm_factor(phi_test, sigma_square=pnml_h.var)
    regret = np.log(nf)
    trainset_size, num_features = data_h.phi_train.shape

    # Analytical pNML
    analytical_nf = analytical_pnml_h.calc_norm_factor(phi_test, sigma_square=pnml_h.var)
    analytical_regret = np.log(analytical_nf)

    return {'x_test': x_test, 'nf': nf, 'regret': regret, 'y_hat_erm': y_hat_erm,
            'analytical_nf': analytical_nf, 'analytical_regret': analytical_regret,
            'nf0': analytical_pnml_h.nf0, 'nf1': analytical_pnml_h.nf1, 'nf2': analytical_pnml_h.nf2,
            'x_bot_square': analytical_pnml_h.x_bot_square,
            'trainset_size': trainset_size, 'num_features': num_features}
