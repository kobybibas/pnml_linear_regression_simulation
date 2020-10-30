import logging
import os
import os.path as osp
import time

import hydra
import numpy as np

from data_utils.synthetic_data_utils import data_type_dict
from experimnet_utils import execute_x_vec
from learner_utils.analytical_pnml_utils import AnalyticalPNML
from learner_utils.pnml_utils import Pnml, PnmlMinNorm
from ray_utils import ray_init

logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name="pnml_polynomial")
def execute_experiment(cfg):
    t0 = time.time()
    logger.info(cfg.pretty())
    out_path = os.getcwd()

    # Move from output directory to src
    os.chdir('../../src')
    logger.info('[cwd out_dir]=[{} {}]'.format(os.getcwd(), out_path))

    # Create trainset
    data_class = data_type_dict[cfg.data_type]
    data_h = data_class(cfg.x_train, cfg.y_train, cfg.model_degree)
    phi_train, y_train = data_h.phi_train, data_h.y

    # Build pNML
    if cfg.pnml_type == 'pnml':
        pnml_h = Pnml(phi_train, y_train, lamb=cfg.lamb, sigma_square=cfg.sigma_square)
    elif cfg.pnml_type == 'pnml_min_norm':
        pnml_h = PnmlMinNorm(cfg.constrain_factor, cfg.pnml_lambda_optim_dict, phi_train, y_train,
                             lamb=0.0, sigma_square=cfg.sigma_square)
    else:
        raise ValueError(f'pnml_type not supported {cfg.pnml_type}')
    analytical_pnml_h = AnalyticalPNML(phi_train, pnml_h.theta_erm)

    # Initialize multiprocess. Every trainset size in a separate process
    ray_init(cfg.num_cpus, cfg.is_local_mode)

    # Execute x_test
    x_test_array = np.arange(cfg['x_test_min'], cfg['x_test_max'], cfg['dx_test']).round(2)
    res_df = execute_x_vec(x_test_array, data_h, pnml_h, analytical_pnml_h)
    res_df['x_train'] = [cfg.x_train] * len(res_df)
    res_df['y_train'] = [cfg.y_train] * len(res_df)

    # Save
    out_file = osp.join(out_path, 'results.csv')
    res_df.to_csv(out_file, index=False)
    logger.info('Finish! in {:.2f} sec. {}'.format(time.time() - t0, out_file))


if __name__ == "__main__":
    execute_experiment()
