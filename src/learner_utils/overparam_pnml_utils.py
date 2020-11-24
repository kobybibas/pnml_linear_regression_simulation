import logging

import numpy as np
import scipy.signal

from learner_utils.learner_helpers import calc_theta_norm
from learner_utils.optimization_utils import fit_norm_constrained_least_squares
from learner_utils.optimization_utils import optimize_pnml_var
from learner_utils.pnml_utils import BasePNML

logger = logging.getLogger(__name__)


class OverparamPNML(BasePNML):
    def __init__(self, pnml_optim_param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.lamb == 0.0
        self.pnml_optim_params = pnml_optim_param
        self.lamb_optim_param = pnml_optim_param['pnml_lambda_optim_param']
        self.var_optim_param = pnml_optim_param['pnml_var_optim_param']

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.norm_constraint = calc_theta_norm(self.theta_erm)

        self.y_interval_min = pnml_optim_param['y_interval_min']
        self.y_interval_max = pnml_optim_param['y_interval_max']
        self.y_interval_num = pnml_optim_param['y_interval_num']

        # The interval for possible y, for creating pdf
        self.is_y_one_sided_interval = pnml_optim_param['is_one_sided_interval']
        self.y_interval = self.create_y_interval()

        # For analytical
        self.theta_mn_P_N_theta_mn = self.calc_trainset_subspace_projection(self.theta_erm)

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray, lamb: float = 0.0) -> np.ndarray:
        assert lamb == 0.0
        theta, lamb, optim_res_dict = fit_norm_constrained_least_squares(phi_arr, y, self.norm_constraint,
                                                                         self.lamb_optim_param['tol_func'],
                                                                         self.lamb_optim_param['max_iter'],
                                                                         self.logger)
        if 'fitted_lamb_list' not in self.intermediate_dict:
            self.intermediate_dict['fitted_lamb_list'] = []
        self.intermediate_dict['fitted_lamb_list'].append(lamb)
        if 'optim_success' not in self.intermediate_dict:
            self.intermediate_dict['optim_success'] = []
        self.intermediate_dict['optim_success'].append(optim_res_dict['success'])
        return theta

    def calc_x_bot_square(self, x_test) -> float:
        x_test = self.convert_to_column_vec(x_test)

        rank = self.rank
        x_bot_square = np.sum((self.u[:, -self.m + rank:].T @ x_test) ** 2)
        return x_bot_square

    def filter_genie_probs(self, probs_of_genies):
        # todo: handling only one sided
        prob_smooth = scipy.signal.medfilt(probs_of_genies[1:], kernel_size=21)
        prob_smooth = np.append(probs_of_genies[0], prob_smooth)

        # Must decrease
        probs_decreasing = np.maximum.accumulate(prob_smooth[::-1])[::-1]
        return probs_decreasing

    def calc_norm_factor(self, x_test: np.array) -> float:
        """
        Calculate normalization factor using numerical integration
        :param x_test: test features to evaluate.
        :return: log normalization factor.
        """
        # Set interval edges
        x_test = self.convert_to_column_vec(x_test)
        self.y_interval = self.create_y_interval(x_test)

        # Calc genies predictions
        thetas = self.calc_genie_thetas(x_test, self.y_interval)
        probs_of_genies = self.calc_probs_of_genies(x_test, self.y_interval, thetas)
        probs_of_genies_smooth = self.filter_genie_probs(probs_of_genies)
        self.intermediate_dict['thetas'] = thetas
        self.intermediate_dict['probs_of_genies'] = probs_of_genies
        self.intermediate_dict['probs_of_genies_smooth'] = probs_of_genies_smooth
        probs_of_genies = probs_of_genies_smooth

        # Integrate to find the pNML normalization factor
        nf = float(np.trapz(probs_of_genies, x=self.y_interval))
        nf = 2 * nf if self.is_y_one_sided_interval is True else nf
        self.intermediate_dict['nf'] = nf
        return nf

    def create_y_interval(self,
                          x_test: np.ndarray = None,
                          y_interval_min: float = None,
                          y_interval_max: float = None,
                          y_interval_num: int = None,
                          is_y_one_sided_interval: bool = True):

        if y_interval_min is None:
            y_interval_min = self.y_interval_min
        if y_interval_max is None:
            y_interval_max = self.y_interval_max
        if y_interval_num is None:
            y_interval_num = self.y_interval_num

        # Set adaptive end
        if x_test is not None:
            y_interval_max = self.find_y_interval_edges(x_test)

        # The interval for possible y, for creating pdf
        y_to_eval = np.append(0, np.logspace(np.log10(y_interval_min),
                                             np.log10(y_interval_max),
                                             y_interval_num))
        if is_y_one_sided_interval is False:
            y_to_eval = np.unique(np.append(y_to_eval, -y_to_eval))

        # shift around the erm
        if x_test is not None:
            y_to_eval = self.shift_y_interval(x_test, self.theta_erm, y_to_eval)
        return y_to_eval

    def find_y_interval_edges(self, x_test: np.ndarray) -> float:
        """
        Find the minimum largest y that gives prob 0
        :param x_test: test sample data
        :return: maximal edge of the label interval to eval
        """
        prob_genie, y_max = 1.0, 1e-1
        while prob_genie > 0.0:
            y_max *= 10
            y_target = self.shift_y_interval(x_test, self.theta_erm, y_max)[None]
            theta_genie = self.calc_genie_thetas(x_test, y_target)

            # Calc genies predictions
            prob_genie = self.calc_probs_of_genies(x_test, y_target, theta_genie)
            prob_genie = float(prob_genie)

        # Extra margin
        y_max *= 100
        return y_max

    def verify_pnml_results(self) -> (bool, str):
        success, msg = True, ''
        nf = self.intermediate_dict['nf']
        probs_of_genies = self.intermediate_dict['probs_of_genies']
        # Some check for converges:
        if nf < 1.0:
            # Expected positive regret
            msg += 'Negative regret. norm_factor={:.6f} '.format(nf)
            success = False
        if probs_of_genies[-1] > np.finfo('float').eps:
            # Expected probability 0 at the edges
            msg += 'Interval is too small prob={}. '.format(probs_of_genies[-1])
            success = False
        # if not np.all(np.diff(probs_of_genies) > 0):
        #     # Expected probability 0 at the edges
        #     msg += 'Genie probs non monotonic. '
        #     success = False
        if success is False:
            y_start, y_end = self.y_interval[0], self.y_interval[-1]
            msg += f'[y_start, y_end]=[{y_start} {y_end}] '
        return success, msg

    @staticmethod
    def shift_y_interval(x_test: np.ndarray, theta_erm: np.ndarray, y_interval: np.ndarray) -> np.ndarray:
        """
        Adapt the y interval to the test sample.
        we want to predict around the ERM prediction based on the analytical result.
        :param x_test: the test sample data.
        :param theta_erm: the erm parameters
        :param y_interval: the basic y interval
        :return: the shifted y interval based on the erm prediction
        """
        y_pred = theta_erm.T @ x_test
        y_vec = y_interval + y_pred
        return np.squeeze(y_vec)

    def calc_genie_thetas(self, x_test: np.ndarray, y_to_eval: np.ndarray) -> (list, list):
        # Each thetas entry is corresponds to entry on y_to_eval
        thetas = []
        y_erm = float(self.theta_erm.T @ x_test)
        for y in y_to_eval:
            if y == y_erm:
                theta = self.theta_erm
            else:
                x_arr, y_vec = self.add_test_to_train(self.x_arr_train, x_test, self.y_vec_train, y)
                theta = self.fit_least_squares_estimator(x_arr, y_vec)
            thetas.append(theta)
        return thetas

    def calc_probs_of_genies(self, x_test, y_trained: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        Calculate the genie probability of the label it was trained with
        :param x_test: test set sample
        :param y_trained: The labels that the genie was trained with
        :param thetas: The fitted parameters to the label (the trained genie)
        :return: the genie probability of the label it was trained with
        """
        var = self.var
        y_hat = np.array([theta.T @ x_test for theta in thetas]).squeeze()
        y_trained = y_trained.squeeze()
        prob_genies = np.exp(-(y_trained - y_hat) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        return prob_genies.squeeze()

    def optimize_var(self, x_test: np.ndarray, y_gt: float) -> (float, float):
        nf = self.calc_norm_factor(x_test)
        success, msg = self.verify_pnml_results()
        if success is False:
            self.intermediate_dict['var_best'] = np.inf
            return -1.0

        # Pre calc optimize sigma
        thetas = self.intermediate_dict['thetas']
        phi_arr, ys = self.add_test_to_train(self.x_arr_train, x_test, self.y_vec_train, y_gt)
        theta_genie_gt = self.fit_least_squares_estimator(phi_arr, ys)

        # Calc best sigma
        y_vec = self.y_interval
        epsilon_square_list = (y_vec - np.array([theta.T @ x_test for theta in thetas]).squeeze()) ** 2
        epsilon_square_true = (y_gt - theta_genie_gt.T @ x_test) ** 2
        var_best = optimize_pnml_var(epsilon_square_true, epsilon_square_list, y_vec, self.var_optim_param)
        self.intermediate_dict['var_best'] = var_best
        return var_best

    def verify_var_results(self) -> (bool, str):
        # Initialize output
        success, msg = True, ''
        var_best = self.intermediate_dict['var_best']

        # Some check for converges:
        if np.isposinf(var_best):
            # Expected positive regret
            msg += 'Infinite var. Did not converge. '
            success = False
        return success, msg

    def calc_trainset_subspace_projection(self, x: np.ndarray) -> float:
        x_projection = np.squeeze(np.power(self.u.T @ x, 2))[:self.rank]
        x_parallel_square = np.sum(x_projection / self.h_square[:self.rank])
        return float(x_parallel_square)

    def calc_analytical_norm_factor(self, x_test):
        # Initialize
        x_test = self.convert_to_column_vec(x_test)
        var = self.var
        rank = self.rank

        # Under param
        # Project on empirical correlation matrix and divide by the associated eigenvalue
        nf0 = 1 + np.sum(np.squeeze(self.u[:, :rank].T @ x_test) ** 2 / self.h_square[:rank])
        self.intermediate_dict['nf0'] = nf0

        # ||x_\bot||^2
        x_bot_square = self.calc_x_bot_square(x_test)

        # Over param
        c = x_bot_square * self.theta_mn_P_N_theta_mn / (np.pi * var)
        if c < 0:
            logger.warning('Lower than zero. x_bot_square={} theta_mn_P_N_theta_mn={}'.format(
                x_bot_square, self.theta_mn_P_N_theta_mn))
        nf1 = float(3 * np.power(c, 1. / 3))
        self.intermediate_dict['nf1'] = nf1

        nf = nf0 * (1 + 2 * x_bot_square) + nf1
        return float(nf)

    def calc_pnml_logloss(self, x_test: np.ndarray, y_gt: float, nf: np.ndarray) -> float:
        # Make the input as column vector
        x_test = self.convert_to_column_vec(x_test)
        var = self.var

        # Add test to train
        x_arr, y_vec = self.add_test_to_train(self.x_arr_train, x_test, self.y_vec_train, y_gt)
        theta_genie = self.fit_least_squares_estimator(x_arr, y_vec, self.lamb)
        logloss = 0.5 * np.log(2 * np.pi * var * (nf ** 2)) + (y_gt - theta_genie.T @ x_test) ** 2 / (
                2 * var)
        return float(logloss)

    def calc_genie_logloss(self, x_test: np.ndarray, y_gt: float, nf: float = 1.0) -> float:
        # Make the input as column vector
        x_test = self.convert_to_column_vec(x_test)
        var = self.var

        # Add test to train
        x_arr, y_vec = self.add_test_to_train(self.x_arr_train, x_test, self.y_vec_train, y_gt)
        theta_genie = self.fit_least_squares_estimator(x_arr, y_vec, self.lamb)
        logloss = 0.5 * np.log(2 * np.pi * var) + (y_gt - theta_genie.T @ x_test) ** 2 / (2 * var)
        return float(logloss)

    def plot_genies_prob(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, sharex=True)
        ax = axs[0]
        ax.plot(self.y_interval, self.intermediate_dict['probs_of_genies'], '*')
        ax.set_xscale('log')
        ax.set_ylabel('Prob')
        ax.grid()

        ax = axs[1]
        norms = [calc_theta_norm(theta_i) for theta_i in self.intermediate_dict['thetas']]
        ax.plot(self.y_interval, norms, '*')
        ax.set_ylabel('||theta||')
        ax.axhline(self.norm_constraint, color='r')
        ax.grid()

        ax = axs[2]
        ax.plot(self.y_interval, self.intermediate_dict['probs_of_genies_smooth'], '*')
        ax.set_ylabel('Prob smooth', '*')
        ax.set_xlabel('y')
        ax.grid()
        plt.show()
