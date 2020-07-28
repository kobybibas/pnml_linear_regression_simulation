import logging
import os

import hydra
import numpy as np

from data_utils import data_type_dict
from experimnet_utils import execute_x_vec
from learner_utils.pnml_min_norm_utils import PnmlMinNorm
from learner_utils.pnml_utils import Pnml

logger = logging.getLogger(__name__)


@hydra.main(config_name='../configs/pnml_min_norm_fourier.yaml') # for pnml_min_norm
# @hydra.main(config_path='../configs/pnml_fourier.yaml')  # for vanilla pnml
def execute_experiment(cfg):
    logger.info(f"Run config:\n{cfg.pretty()}")
    logger.info(os.getcwd())

    # Create trainset
    data_class = data_type_dict[cfg.data_type]
    data_h = data_class(cfg.x_train, cfg.y_train, cfg.model_degree)
    phi_train, y_train = data_h.create_train_features(), data_h.y

    # Build pNML
    if cfg.pnml_type == 'pnml':
        pnml_h = Pnml(phi_train, y_train, lamb=cfg.lamb, var=cfg.min_sigma_square)
    elif cfg.pnml_type == 'pnml_min_norm':
        pnml_h = PnmlMinNorm(cfg.constrain_factor, phi_train, y_train, lamb=0.0, min_sigma_square=cfg.min_sigma_square)
    else:
        raise ValueError(f'pnml_type not supported {cfg.pnml_type}')
    pnml_h.set_y_interval(cfg.y_min, cfg.y_max, cfg.y_num, is_adaptive=cfg.is_adaptive)

    # Execute x_test
    x_test_array = np.arange(cfg['x_test_min'], cfg['x_test_max'], cfg['dx_test']).round(2)
    execute_x_vec(x_test_array, data_h, pnml_h, out_dir=os.getcwd(), is_mp=cfg.is_multi_process)
    logger.info('Finished. Save to: {}'.format(os.getcwd()))
    logger.info('rsync -chavzP --stats aws_cpu:{} .'.format(os.getcwd()))


if __name__ == "__main__":
    execute_experiment()
