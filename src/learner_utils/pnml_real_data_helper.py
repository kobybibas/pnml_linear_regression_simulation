import logging
import time

import numpy as np
import pandas as pd

from learner_utils.learner_helpers import fit_least_squares_estimator, calc_best_var
from learner_utils.overparam_pnml_utils import OverparamPNML
from learner_utils.underparam_pnml_utils import UnderparamPNML

logger_default = logging.getLogger(__name__)


def choose_pnml_h_type(pnml_handlers, x_i, x_bot_threshold):
    overparam_pnml_h = pnml_handlers['overparam']
    underparam_pnml_h = pnml_handlers['underparam']
    assert isinstance(overparam_pnml_h, OverparamPNML)
    assert isinstance(underparam_pnml_h, UnderparamPNML)

    x_norm_square = np.linalg.norm(x_i) ** 2
    if overparam_pnml_h.rank >= overparam_pnml_h.m:
        pnml_h = underparam_pnml_h
        x_bot_square = 0.0
    else:
        x_bot_square = overparam_pnml_h.calc_x_bot_square(x_i)
        pnml_h = overparam_pnml_h if x_bot_square / x_norm_square > x_bot_threshold else underparam_pnml_h
    return pnml_h, x_bot_square, x_norm_square


def optimize_pnml_var_on_valset(pnml_handlers: dict, x_val, y_val, split_num: int,
                                x_bot_threshold=np.finfo('float').eps,
                                logger=logger_default) -> float:
    overparam_pnml_h = pnml_handlers['overparam']
    underparam_pnml_h = pnml_handlers['underparam']
    assert isinstance(overparam_pnml_h, OverparamPNML)
    assert isinstance(underparam_pnml_h, UnderparamPNML)

    # Compute best variance using validation set
    var_best_list = []
    for i, (x_i, y_i) in enumerate(zip(x_val, y_val)):
        t0 = time.time()
        nf, loss = -1, -1

        # Check projection on orthogonal subspace
        pnml_h, x_bot_square, x_norm_square = choose_pnml_h_type(pnml_handlers, x_i, x_bot_threshold)

        # Find best sigma square
        pnml_h.reset()
        var_best = pnml_h.optimize_var(x_i, y_i)
        success, msg = pnml_h.verify_var_results()

        # Further check if optimize succeed
        if success is True:
            pnml_h.var = var_best
            nf = pnml_h.calc_norm_factor(x_i)
            success, msg = pnml_h.verify_pnml_results()
            loss = pnml_h.calc_pnml_logloss(x_i, y_i, nf) if success else -1

            if success is True:
                var_best_list.append(var_best)

        # msg
        log = '[{:03d}/{}] pNML Val: split={:02d} shape={}\t '.format(
            i, len(x_val) - 1, split_num, pnml_h.x_arr_train.shape)
        log += '[nf loss var]=[{:.6f} {:.6f} {:.6f}] '.format(
            nf, loss, pnml_h.var)
        log += '[|x|^2 |x_bot|^2]=[{:.6f} {:.6f}] '.format(
            x_norm_square, x_bot_square)
        log += 'success={}. {:.2f}s. {}'.format(success, time.time() - t0, msg)
        if success is True:
            logger.info(log)
        else:
            logger.warning(log)

    var_best_mean = np.mean(var_best_list)
    return var_best_mean


def calc_pnml_testset_performance(pnml_handlers: dict, x_test: np.ndarray, y_test: np.ndarray,
                                  split_num: int,
                                  x_bot_threshold=np.finfo('float').eps,
                                  logger=logger_default) -> pd.DataFrame:
    overparam_pnml_h = pnml_handlers['overparam']
    underparam_pnml_h = pnml_handlers['underparam']
    assert isinstance(overparam_pnml_h, OverparamPNML)
    assert isinstance(underparam_pnml_h, UnderparamPNML)

    nfs, losses, success_list = [], [], []
    loss_genies, x_bot_square_list, x_norm_square_list = [], [], []
    analytical_nfs = []
    for i, (x_i, y_i) in enumerate(zip(x_test, y_test)):
        t0 = time.time()

        # Check projection on orthogonal subspace
        pnml_h, x_bot_square, x_norm_square = choose_pnml_h_type(pnml_handlers, x_i, x_bot_threshold)
        x_norm_square_list.append(x_norm_square)

        # calc normalization factor (nf)
        pnml_h.reset()
        nf = pnml_h.calc_norm_factor(x_i)
        loss = pnml_h.calc_pnml_logloss(x_i, y_i, nf)
        success, msg = pnml_h.verify_pnml_results()
        nfs.append(nf)
        x_bot_square_list.append(x_bot_square)
        losses.append(loss)
        success_list.append(success)

        # Genie performance
        loss_genie = pnml_h.calc_genie_logloss(x_i, y_i, nf)
        loss_genies.append(loss_genie)
        if loss_genie > -np.log(np.finfo('float').eps):
            msg += 'Genie did not converge. '.format(loss_genie)
            success = False

        # analytical pnml
        analytical_nf = pnml_h.calc_analytical_norm_factor(x_i)
        analytical_nfs.append(analytical_nf)
        # msg
        log = '[{:03d}/{}] pNML Test: split={:02d} shape={}\t '.format(
            i, len(x_test) - 1, split_num, pnml_h.x_arr_train.shape)
        log += '[nf loss var]=[{:.6f} {:.6f} {:.6f}] '.format(
            nf, loss, pnml_h.var)
        log += '[|x|^2 |x_bot|^2]=[{:.6f} {:.6f}] '.format(
            x_norm_square, x_bot_square)
        log += 'success={}. {:.2f}s. {}'.format(success, time.time() - t0, msg)
        if success is True:
            logger.info(log)
        else:
            logger.warning(log)

    # Add to dict
    res_dict = {'pnml_regret': np.log(nfs).tolist(),
                'pnml_test_logloss': losses,
                'pnml_variance': [pnml_h.var] * len(x_test),
                'pnml_success': success_list,
                'pnml_norm_factors': nfs,
                'x_bot_square': x_bot_square_list,
                'x_norm_square': x_norm_square_list,
                'genie_test_logloss': loss_genies,
                'analytical_pnml_regret': np.log(analytical_nfs)}
    df = pd.DataFrame(res_dict)
    return df


def calc_pnml_performance(x_train: np.ndarray, y_train: np.ndarray,
                          x_val: np.ndarray, y_val: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray,
                          split_num: int,
                          pnml_optim_param: dict,
                          logger=logger_default) -> pd.DataFrame:
    # Initialize pNML
    pnml_handlers = {'underparam': UnderparamPNML(x_arr_train=x_train, y_vec_train=y_train,
                                                  var=pnml_optim_param['var'], lamb=0.0, logger=logger),
                     'overparam': OverparamPNML(x_arr_train=x_train, y_vec_train=y_train,
                                                var=pnml_optim_param['var'], lamb=0.0, logger=logger,
                                                pnml_optim_param=pnml_optim_param)}

    # Initialize pNML with ERM var.
    if pnml_optim_param['var'] <= 0.0:
        theta_erm = fit_least_squares_estimator(x_train, y_train)
        var = calc_best_var(x_val, y_val, theta_erm)
        pnml_handlers['underparam'].var = var
        pnml_handlers['underparam'].var_input = var
        pnml_handlers['overparam'].var = var
        pnml_handlers['overparam'].var_input = var
    x_bot_threshold = pnml_optim_param['x_bot_threshold']

    # Compute best variance using validation set
    valset_mean_var = var
    if not pnml_optim_param["skip_pnml_optimize_var"]:
        valset_mean_var = optimize_pnml_var_on_valset(pnml_handlers, x_val, y_val, split_num, x_bot_threshold, logger)

    # Assign best var
    pnml_handlers['underparam'].var = valset_mean_var
    pnml_handlers['underparam'].var_input = valset_mean_var
    pnml_handlers['overparam'].var = valset_mean_var
    pnml_handlers['overparam'].var_input = valset_mean_var

    # Execute on test set
    df = calc_pnml_testset_performance(pnml_handlers, x_test, y_test, split_num,
                                       x_bot_threshold,
                                       logger)
    return df
