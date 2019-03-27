import pandas as pd
from tqdm import tqdm

from data_utilities import DataParameters
from data_utilities import DataPolynomial
from general_utilies import *
from pnml_utilities import Pnml
from pnml_utilities import PnmlNoSigma
from pnml_utilities import PNMLParameters


# from pnml_utilities import PNMLwoSigma as PNML


class ExperimentParameters:
    def __init__(self):
        self.x_test_max = 5.0
        self.dx_test = 0.1

        self.poly_degree_list = [1, 2, 3, 4]

        self.pnml_type = 'vanilla'
        self.pnml_possible_types = ['vanilla', 'no_sigma']

    def __str__(self):
        string = 'ExperimentParameters:\n'
        string += '    x_test_max: {}\n'.format(self.x_test_max)
        string += '    dx_test: {}\n'.format(self.dx_test)
        string += '    poly_degree_list: {}\n'.format(self.poly_degree_list)
        string += '    pnml_type: {}\n'.format(self.pnml_type)
        string += '    pnml_possible_types: {}\n'.format(self.pnml_possible_types)
        return string


class Experiment:
    def __init__(self,
                 exp_params: ExperimentParameters,
                 data_params: DataParameters,
                 pnml_params: PNMLParameters):

        self.data_params = data_params
        self.pnml_params = pnml_params
        self.exp_params = exp_params

        # Create interval
        self.x_test_array = np.arange(-exp_params.x_test_max,
                                      exp_params.x_test_max,
                                      exp_params.dx_test).round(2)

        self.exp_df_dict = {}
        self.regret_df = pd.DataFrame(columns=exp_params.poly_degree_list,
                                      index=self.x_test_array)

        self.x_train = None
        self.y_train = None

    def execute(self):

        data_h = DataPolynomial(self.data_params)
        if self.exp_params.pnml_type == 'vanilla':
            pnml_h = Pnml()
        if self.exp_params.pnml_type == 'no_sigma':
            pnml_h = PnmlNoSigma()
        for poly_degree in self.exp_params.poly_degree_list:
            self.exp_df_dict[str(poly_degree)] = self.execute_poly_degree(data_h, pnml_h, poly_degree)

    def execute_poly_degree(self, data_h, pnml_h, poly_degree):

        # Create training set
        data_h.create_train(poly_degree)
        self.x_train, self.y_train = data_h.get_data_points_as_list()

        # Initialize experiment df
        y_to_eval = pnml_h.get_y_interval(self.pnml_params.y_max, self.pnml_params.dy)
        exp_df = pd.DataFrame(columns=self.x_test_array, index=y_to_eval)

        # Iterate on test points
        for x_test in tqdm(self.x_test_array):
            predictor, regret = self.execute_x_test(data_h, pnml_h, x_test)
            exp_df[x_test] = predictor
            self.regret_df[poly_degree][x_test] = regret
        return exp_df

    def execute_x_test(self, data_h, pnml_h, x_test):
        pnml_h.compute_predictor(x_test, self.pnml_params, data_h)
        return pnml_h.get_predictor(), pnml_h.regret

    def get_train(self):
        return self.x_train, self.y_train

    def get_regret_df(self):
        return self.regret_df

    def get_exp_df_dict(self):
        return self.exp_df_dict
