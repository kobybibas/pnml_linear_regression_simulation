from tqdm import tqdm

from data_utilities import DataParameters
from data_utilities import DataPolynomial
from general_utilies import *
from pnml_joint_utilities import PnmlJoint
from pnml_utilities import PNMLParameters
from pnml_utilities import Pnml


class ExperimentParameters:
    def __init__(self):
        self.experiment_name = 'pnml_vanilla'

        self.output_dir_base = '../output/figures'

        self.x_test_max = 3.0
        self.x_test_min = -self.x_test_max

        self.dx_test = 0.1

        # Polynomial degree to evaluate
        self.poly_degree_list = [1, 2, 3]

        # Regularization term to evaluate
        self.lambda_list = [0.0, 0.1, 1.0]

        # pnml learner type
        self.pnml_type = 'only_theta'
        self.pnml_possible_types = ['only_theta',  # using variance of the ERM: sigma^2=\sum (y-\hat{y})^2
                                    'joint']  # jointly eval least squares parameters and the variance.

        # The type of variable to iterate on
        self.exp_type = 'poly'
        self.exp_possible_types = ['poly',
                                   'lambda']

        # Avilable exp types

    def __str__(self):
        string = 'ExperimentParameters:\n'
        string += '    output_dir_base: {}\n'.format(self.output_dir_base)
        string += '    x_test_max: {}\n'.format(self.x_test_max)
        string += '    x_test_max: {}\n'.format(self.x_test_min)

        string += '    dx_test: {}\n'.format(self.dx_test)
        string += '    poly_degree_list: {}\n'.format(self.poly_degree_list)
        string += '    lambda_list: {}\n'.format(self.lambda_list)
        string += '    pnml_type: {}\n'.format(self.pnml_type)
        string += '    pnml_possible_types: {}\n'.format(self.pnml_possible_types)
        string += '    pnml_type: {}\n'.format(self.exp_type)
        string += '    pnml_possible_types: {}\n'.format(self.exp_possible_types)
        return string


class Experiment:
    def __init__(self,
                 exp_params: ExperimentParameters,
                 data_params: DataParameters,
                 pnml_params: PNMLParameters):

        self.data_params = data_params
        self.pnml_params = pnml_params
        self.exp_params = exp_params

        # Create intervals of test points.
        self.x_test_array = np.arange(exp_params.x_test_min,
                                      exp_params.x_test_max,
                                      exp_params.dx_test).round(2)

        # Output dictionary
        self.exp_df_dict = {}

        # The log(normalization factor) dataframe.
        self.regret_df = pd.DataFrame(columns=exp_params.poly_degree_list,
                                      index=self.x_test_array)

        # Training data
        self.x_train = None
        self.y_train = None

    def execute_poly_degree_search(self):
        """
        Evaluate pnml learner for each polynomial degree in self.exp_params.poly_degree_list
        :return:
        """

        # Initialize output
        self.regret_df = pd.DataFrame(columns=self.exp_params.poly_degree_list,
                                      index=self.x_test_array)

        data_h = DataPolynomial(self.data_params)
        if self.exp_params.pnml_type == 'only_theta':
            pnml_h = Pnml()
        elif self.exp_params.pnml_type == 'joint':
            pnml_h = PnmlJoint()
        else:
            ValueError('pnml type doesnt exist')

        # Iterate on polynomial degree
        for poly_degree in self.exp_params.poly_degree_list:
            self.exp_df_dict[str(poly_degree)] = self.execute_poly_degree(data_h, pnml_h, poly_degree)

    def execute_lambda_search(self):
        """
        Evaluate pnml learner for each regularization term in self.exp_params.lambda_list
        :return:
        """

        # Initialize output
        self.regret_df = pd.DataFrame(columns=self.exp_params.lambda_list,
                                      index=self.x_test_array)

        data_h = DataPolynomial(self.data_params)
        if self.exp_params.pnml_type == 'only_theta':
            pnml_h = Pnml()
        elif self.exp_params.pnml_type == 'joint':
            pnml_h = PnmlJoint()
        else:
            ValueError('pnml type doesnt exist')

        # Iterate on regularization term
        for lamb in self.exp_params.lambda_list:
            self.exp_df_dict[str(lamb)] = self.execute_lambda(data_h, pnml_h,
                                                              self.data_params.poly_degree, lamb)

    def execute_lambda(self, data_h: DataPolynomial, pnml_h: Pnml, poly_degree: int, lamb: float) -> pd.DataFrame:
        """
        Evaluate the pnml learner for specific regularization term
        :param data_h: DataPolynomial class handler.
        :param pnml_h: Pnml class handler.
        :param poly_degree: the assumed polynomial degree of the data.
        :param lamb: the regularization term.
        :return: data frame of the experiment. column: x_test, rows: y probability.
        """

        # Create training set
        data_h.create_train(poly_degree)
        self.x_train, self.y_train = data_h.get_data_points_as_list()

        # Initialize experiment df
        y_to_eval = pnml_h.get_y_interval(self.pnml_params.y_min, self.pnml_params.y_max, self.pnml_params.dy)
        exp_df = pd.DataFrame(columns=self.x_test_array, index=y_to_eval)

        # Iterate on test points
        for x_test in tqdm(self.x_test_array):
            predictor, regret = self.execute_x_test(data_h, pnml_h, x_test, lamb)
            exp_df[x_test] = predictor
            self.regret_df[lamb][x_test] = regret
        return exp_df

    def execute_poly_degree(self, data_h: DataPolynomial, pnml_h: Pnml, poly_degree: int) -> pd.DataFrame:
        """
        Evaluate the pnml learner for specific polynomial degree.
        :param data_h: DataPolynomial class handler.
        :param pnml_h: Pnml class handler.
        :param poly_degree: the assumed polynomial degree of the data.
        :param lamb: the regularization term.
        :return: data frame of the experiment. column: x_test, rows: y probability.
        """

        # Create training set
        data_h.create_train(poly_degree)
        self.x_train, self.y_train = data_h.get_data_points_as_list()

        # Initialize experiment df
        y_to_eval = pnml_h.get_y_interval(self.pnml_params.y_min, self.pnml_params.y_max, self.pnml_params.dy)
        exp_df = pd.DataFrame(columns=self.x_test_array, index=y_to_eval)

        # Iterate on test points
        for x_test in tqdm(self.x_test_array):
            predictor, regret = self.execute_x_test(data_h, pnml_h, x_test, self.data_params.lamb)
            exp_df[x_test] = predictor
            self.regret_df[poly_degree][x_test] = regret
        return exp_df

    def execute_x_test(self, data_h, pnml_h, x_test, lamb=1e-4):
        pnml_h.compute_predictor(x_test, self.pnml_params, data_h, lamb)
        return pnml_h.get_predictor(), pnml_h.regret

    def get_train(self):
        return self.x_train, self.y_train

    def get_regret_df(self):
        return self.regret_df

    def get_exp_df_dict(self):
        return self.exp_df_dict
