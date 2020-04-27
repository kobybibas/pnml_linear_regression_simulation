import logging
import os

import hydra
import numpy as np

from data_utils import data_type_dict
from experimnet_utils import execute_x_vec
from pnml_min_norm_utils import PnmlMinNorm

logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs/min_norm_fourier.yaml')
def execute_experiment(cfg):
    logger.info(f"Run config:\n{cfg.pretty()}")

    # Create trainset
    DataC = data_type_dict[cfg.data_type]
    data_h = DataC(cfg.x_train, cfg.y_train, cfg.model_degree)
    phi_train, y_train = data_h.create_train_features(), data_h.y

    # Build pNML
    pnml_h = PnmlMinNorm(cfg.constrain_factor, phi_train, y_train, lamb=0.0, min_sigma_square=cfg.min_sigma_square)
    pnml_h.set_y_interval(cfg.y_min, cfg.y_max, cfg.dy, is_adaptive=cfg.is_adaptive)

    # Execute x_test
    x_test_array = np.arange(cfg['x_test_min'], cfg['x_test_max'], cfg['dx_test']).round(2)
    execute_x_vec(x_test_array, data_h, pnml_h, is_mp=cfg.is_multi_process, out_dir=os.getcwd())
    logger.info('Finished. Save to: {}'.format(os.getcwd()))
    logger.info('rsync -chavzP --stats aws_cpu:{} .'.format(os.getcwd()))


if __name__ == "__main__":
    execute_experiment()
