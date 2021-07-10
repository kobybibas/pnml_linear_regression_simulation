import logging

import psutil
import ray

logger = logging.getLogger(__name__)


def ray_init(cpu_num: int, is_local_mode: bool):
    """
    Initialize multiprocess. Every trainset size in a separate process
    :param cpu_num: NUmber of cpu to use. -1 to use all available
    :param is_local_mode: whether to run in the local computer. Used for debug.
    :return:
    """
    cpu_count = psutil.cpu_count()
    cpu_to_use = cpu_num if cpu_num > 0 else cpu_count
    cpu_to_use = cpu_to_use if is_local_mode is False else 0
    logger.info(f'{cpu_count=}. Executing on {cpu_to_use}. {is_local_mode=}')
    ray.init(local_mode=is_local_mode, num_cpus=cpu_to_use)
