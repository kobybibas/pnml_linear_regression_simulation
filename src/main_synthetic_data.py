import logging
import os
import os.path as osp
import time

import hydra
import numpy as np

from data_utils.synthetic_data_utils import data_type_dict
from experimnet_utils import execute_x_vec
from learner_utils.overparam_pnml_utils import OverparamPNML
from learner_utils.underparam_pnml_utils import UnderparamPNML
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
    x_train, y_train = data_h.phi_train, data_h.y

    # Build pNML
    pnml_handlers = {'underparam': UnderparamPNML(x_arr_train=x_train, y_vec_train=y_train,
                                                  var=cfg.pnml_optim_param['var'], lamb=cfg.lamb, logger=logger),
                     'overparam': OverparamPNML(x_arr_train=x_train, y_vec_train=y_train,
                                                var=cfg.pnml_optim_param['var'], lamb=cfg.lamb, logger=logger,
                                                pnml_optim_param=cfg.pnml_optim_param)}

    # Initialize multiprocess. Every trainset size in a separate process
    ray_init(cfg.num_cpus, cfg.is_local_mode)

    # Execute x_test
    x_test = np.arange(cfg['x_test_min'], cfg['x_test_max'], cfg['dx_test']).round(3)

    res_df = execute_x_vec(x_test, data_h, pnml_handlers, cfg.pnml_optim_param.x_bot_threshold)
    res_df['x_train'] = [cfg.x_train] * len(res_df)
    res_df['y_train'] = [cfg.y_train] * len(res_df)

    # Save
    out_file = osp.join(out_path, 'results.csv')
    res_df.to_csv(out_file, index=False)
    logger.info('Finish! in {:.2f} sec. {}'.format(time.time() - t0, out_file))


if __name__ == "__main__":
    execute_experiment()
