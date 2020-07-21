import logging
import time

import numpy as np
import scipy.optimize

from learner_classes.learner_utils import estimate_sigma_with_valset

logger = logging.getLogger(__name__)


def add_test_to_train(phi_train: np.ndarray, phi_test: np.ndarray) -> np.ndarray:
    """
    Add the test set feature to training set feature matrix
    :param phi_train: training set feature matrix.
    :param phi_test: test set feature.
    :return: concat train and test.
    """
    # Make the input as row vector
    if len(phi_test.shape) == 1:
        phi_test = np.expand_dims(phi_test, 0)
    phi_arr = np.concatenate((phi_train, phi_test), axis=0)
    return phi_arr


def fit_least_squares_estimator(phi_arr: np.ndarray, y: np.ndarray, lamb: float = 0.0) -> np.ndarray:
    """
    Fit least squares estimator
    :param phi_arr: The training set features matrix. Each row represents an example.
    :param y: the labels vector.
    :param lamb: regularization term.
    :return: the fitted parameters.
    """
    phi_t_phi = phi_arr.T @ phi_arr
    inv = np.linalg.pinv(phi_t_phi + lamb * np.eye(phi_t_phi.shape[0], phi_t_phi.shape[1]))
    theta = inv @ phi_arr.T @ y
    theta = np.expand_dims(theta, 1)
    return theta


def compute_pnml_logloss(x_arr: np.ndarray, y_true: np.ndarray, theta_genies: np.ndarray, var: float,
                         nfs: np.ndarray) -> float:
    y_hat = np.array([x @ theta_genie for x, theta_genie in zip(x_arr, theta_genies)]).squeeze()
    prob = np.exp(-(y_hat - y_true) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    # Normalize by the pnml normalization factor
    prob /= nfs
    logloss = -np.log(prob + np.finfo('float').eps)
    return logloss.mean()


def calc_theta_norm(theta: np.ndarray):
    return (theta ** 2).mean()


def fit_least_squares_with_max_norm_constrain(phi_arr: np.ndarray, y: np.ndarray, max_norm: float,
                                              minimize_dict: dict = None,
                                              theta_0: np.ndarray = None) -> np.ndarray:
    """
    Fit least squares estimator. Constrain it by the max norm constrain
    :param phi_arr: data matrix. Each row represents an example.
    :param y: label vector
    :param max_norm: the constraint of the fitted parameters.
    :param minimize_dict: configuration for minimization function.
    :param theta_0: the initial guess of the learned parameters.
    :return: fitted parameters that satisfy the max norm constrain
    """
    n, m = phi_arr.shape
    assert n == len(y)

    # apply default minimization params
    if minimize_dict is None:
        minimize_dict = {'options': {'disp': False, 'maxiter': 100000, 'ftol': 1e-12}}

    def minimization_constrain(lamb_fit):
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        return calc_theta_norm(theta_fit)

    def mse(lamb_fit):
        theta_fit = fit_least_squares_estimator(phi_arr, y, lamb=lamb_fit)
        return np.power(y - phi_arr @ theta_fit, 2).mean()

    # Initial gauss
    lamb_0 = 1.0

    # Optimize
    nonlinear_constraint = scipy.optimize.NonlinearConstraint(minimization_constrain, 0, max_norm)
    res = scipy.optimize.minimize(mse, lamb_0, constraints=[nonlinear_constraint], options=minimize_dict['options'])
    lamb = res.x
    theta = fit_least_squares_estimator(phi_arr, y, lamb=lamb)

    if res.success == False:
        logger.warning('fit_least_squares_with_max_norm_constrain: Failed')
        logger.warning('y: {}. [||theta||^2 max_norm]= [{} {}]'.format(y[-1], calc_theta_norm(theta), max_norm))
        logger.warning(res)
    return theta


class Pnml:
    def __init__(self, phi_train: np.ndarray, y_train: np.ndarray, lamb: float = 0.0, var: float = 1e-6):
        # Minimum variance for prediction
        self.var = var

        # dict of the genies output:
        #     y_vec: labels that were evaluated, probs:probability of the corresponding labels
        self.genies_output = {'y_vec': None, 'probs': None, 'epsilons': None, 'theta_genie': None}
        self.norm_factor = None

        # The interval for possible y, for creating pdf
        self.y_to_eval = np.arange(-1000, 1000, 0.01)

        # Train feature matrix and labels
        self.phi_train = phi_train
        self.y_train = y_train

        # Regularization term
        self.lamb = lamb

        # ERM least squares parameters
        self.theta_erm = None

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray):
        return fit_least_squares_estimator(phi_arr, y, lamb=self.lamb)

    def set_y_interval(self, dy_min: float, y_max: float, y_num: int, is_adaptive: bool = False, logbase=10):
        """
        Build list on which the probability will be evaluated.
        :param dy_min: first evaluation point after 0.
        :param y_max: the higher bound of the interval.
        :param y_num: number of points to evaluate
        :param is_adaptive: gives more dy around zero.
        :return: y list on which to eval the probability.
        """
        # Initialize evaluation interval
        assert dy_min > 0

        y_to_eval = np.append([0], np.logspace(np.log(dy_min) / np.log(logbase),
                                               np.log(y_max) / np.log(logbase), int(y_num / 2), base=logbase))
        y_to_eval = np.unique(np.concatenate((-y_to_eval, y_to_eval)))

        if is_adaptive is True:
            y_to_eval = np.concatenate((np.arange(0, 0.001, 1e-7),
                                        np.arange(0.001, 1, 1e-3),
                                        np.arange(1.0, 10, 0.1),
                                        np.arange(10, 100, 1.0),
                                        np.arange(100, 1000, 10.0)))
            y_to_eval = np.unique(np.concatenate((-y_to_eval, y_to_eval)))
        self.y_to_eval = y_to_eval

    def predict_erm(self, phi_test: np.ndarray) -> float:
        if self.theta_erm is None:
            self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.y_train)
        return float(self.theta_erm.T @ phi_test)

    @staticmethod
    def calc_genies_probs(y_trained: np.ndarray, y_hat: np.ndarray, var: float) -> np.ndarray:
        """
        Calculate the genie probability of the label it was trained with
        :param y_trained: The labels that the genie was trained with
        :param y_hat: The predicted label by the trained genie
        :param var: the variance (sigma^2)
        :return: the genie probability of the label it was trained with
        """
        genies_probs = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(y_trained - y_hat) ** 2 / (2 * var))
        return genies_probs

    def execute_regret_calc(self, phi_test: np.array, y_test_true: float = None) -> float:
        """
        Calculate normalization factor using numerical integration
        :param phi_test: test features to evaluate.
        :param y_test_true: the true test label. If supplied return the genie params.
        :return: log normalization factor.
        """
        if self.theta_erm is None:
            self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.y_train)

        phi_arr = add_test_to_train(self.phi_train, phi_test)

        # Predict around the ERM prediction
        y_pred = self.theta_erm.T @ phi_test
        y_vec = self.y_to_eval + y_pred

        # Add the true test label to the evaluation vec
        y_test_idx = None
        if y_test_true is not None:
            y_to_eval = np.sort(np.unique(np.append(y_vec, y_test_true)))
            y_test_idx = int(np.where(y_to_eval == y_test_true)[0])

        # Calc genies predictions
        thetas = [self.fit_least_squares_estimator(phi_arr, np.append(self.y_train, y)) for y in y_vec]
        y_hats = np.array([theta.T @ phi_test for theta in thetas]).squeeze()
        genies_probs = self.calc_genies_probs(y_vec, y_hats, self.var)

        # Integrate to find the pNML normalization factor
        norm_factor = np.trapz(genies_probs, x=y_vec)

        # The genies predictors
        self.genies_output = {'y_vec': y_vec, 'probs': genies_probs, 'epsilons': y_vec - y_hats,
                              # Get theta genie
                              'theta_genie': thetas[y_test_idx] if y_test_true is not None else None}
        self.norm_factor = norm_factor
        regret = np.log(norm_factor)
        return regret


class PnmlMinNorm(Pnml):
    def __init__(self, constrain_factor, *args, **kargs):
        # Initialize all other class var
        super().__init__(*args, **kargs)

        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

        # Fitted least squares parameters with norm constraint
        self.lamb = 0.0
        self.theta_erm = fit_least_squares_estimator(self.phi_train, self.y_train)
        self.max_norm = self.constrain_factor * calc_theta_norm(self.theta_erm)

    def set_constrain_factor(self, constrain_factor: float):
        # The norm constrain is set to: constrain_factor * ||\theta_MN||^2
        self.constrain_factor = constrain_factor

    def fit_least_squares_estimator(self, phi_arr: np.ndarray, y: np.ndarray) -> np.ndarray:
        max_norm = self.max_norm
        theta = fit_least_squares_with_max_norm_constrain(phi_arr, y, max_norm,
                                                          minimize_dict=None, theta_0=self.theta_erm)
        return theta


def calc_empirical_pnml_performance(x_train: np.ndarray, y_train: np.ndarray,
                                    x_val: np.ndarray, y_val: np.ndarray,
                                    x_test: np.ndarray, y_test: np.ndarray,
                                    theta_erm: np.ndarray = None,
                                    theta_genies_out=None) -> (dict, np.ndarray):
    n_train, num_features = x_train.shape

    # Fit ERM
    if theta_erm is None:
        theta_erm = fit_least_squares_estimator(x_train, y_train, lamb=0.0)
    var = estimate_sigma_with_valset(x_val, y_val, theta_erm)

    # Empirical pNML
    if n_train > num_features:
        pnml_h = Pnml(phi_train=x_train, y_train=y_train, lamb=0.0, var=var)
    else:
        pnml_h = PnmlMinNorm(constrain_factor=1.0, phi_train=x_train, y_train=y_train, lamb=0.0, var=var)

    y_max = 20 * (max(y_train.max(), y_test.max()) - min(y_train.min(), y_test.min()))
    y_min, y_num, is_adaptive = 1e-9, 60, False  # todo: based on data
    pnml_h.set_y_interval(y_min, y_max, y_num, is_adaptive=is_adaptive)
    regrets, norm_factors, theta_genies = [], [], []
    for j, (x_test_i, y_test_i) in enumerate(zip(x_test, y_test)):
        t0 = time.time()
        regret = pnml_h.execute_regret_calc(x_test_i, y_test_i)
        norm_factor = pnml_h.norm_factor
        regrets.append(regret)
        norm_factors.append(norm_factor)
        theta_genies.append(pnml_h.genies_output['theta_genie'])

        # Some check for converges:
        if regret < 0:
            # Expected positive regret
            logger.warning('Warning. regret is not valid. {}'.format(regret))
        if pnml_h.genies_output['probs'][-1] > np.finfo('float').eps:
            # Expected probability 0 at the edges
            logger.warning('Warning. Interval is too small. prob={}'.format(pnml_h.genies_output['probs'][-1]))
        logger.info('[{}/{}] y_test_i={} regret={:.2f} in {:.2f} sec'.format(j, len(y_test),
                                                                             y_test_i, regret, time.time() - t0))

        if False:
            import matplotlib.pyplot as plt
            plt.plot(pnml_h.genies_output['y_vec'], pnml_h.genies_output['probs'], '*')
            plt.axvline(x=y_test_i, label='True', color='g')
            plt.axvline(x=theta_erm.T @ x_test_i, label='ERM', color='r')
            plt.legend()
            # plt.xscale('log')
            plt.show()

    res_dict = {'regret': np.mean(regrets),
                'test_logloss': compute_pnml_logloss(x_test, y_test, theta_genies, var, norm_factors)}

    # Fill output
    if theta_genies_out is not None:
        theta_genies_out.append(theta_genies)
    return res_dict
