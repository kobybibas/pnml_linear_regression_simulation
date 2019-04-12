import os

import matplotlib.pyplot as plt

from data_utilities import DataParameters
from experimnet_utilities import Experiment, ExperimentParameters
from pnml_utilities import PNMLParameters
from pnml_utilities import get_argmax_prediction

output_dir = '../output/isit/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create parameters
data_params = DataParameters()
pnml_params = PNMLParameters()
exp_params = ExperimentParameters()
data_params.num_points = 10
exp_params.poly_degree_list = [3, 6, 10]
data_params.x_train = [-0.6, -0.55, 0.1, 0.15, 0.2, 0.4, 0.45, 0.55, 0.6, 0.9]
data_params.y_train = [0.5, 0.3, 0.55, 0.4, 0.6, -0.2, 0.1, -0.35, -0.7, 0.2]
exp_params.lambda_list = [0.0, 0.1, 1.0]
pnml_params.y_max = 200
pnml_params.y_min = -200
pnml_params.dy = 0.001
exp_params.x_test_max = 1.1
exp_params.dx_test = 0.05
exp_params.pnml_type = 'only_theta'
print(data_params)
print(pnml_params)
print(exp_params)

# Execute poly degree
exp_h = Experiment(exp_params, data_params, pnml_params)
exp_h.execute_poly_degree_search()
regret_df = exp_h.get_regret_df()
exp_df_dict = exp_h.get_exp_df_dict()
x_train, y_train = exp_h.get_train()

# Get argmax predictor
argmax_prediction_dict = get_argmax_prediction(exp_df_dict)

#################
# Plots
plt.style.use('seaborn-dark')
size = plt.figure().get_size_inches()
plt.figure(figsize=(size[0], size[1] * 2))

# Plot argmax prediction
fig, axs = plt.subplots(2, 1)
for poly_degree, prediction in argmax_prediction_dict.items():
    axs[0].plot(exp_h.x_test_array, prediction, label='Poly Degree = %s' % poly_degree)
axs[0].set_ylabel('Label')
axs[0].grid(True)
axs[0].plot(x_train, y_train, 'o', label='train', markersize=4)
axs[0].set_xlim(-exp_params.x_test_max, exp_params.x_test_max)
axs[0].set_ylim(-8, 15)
axs[0].set_xlabel('')
axs[0].legend()
plt.setp(axs[0].get_xticklabels(), visible=False)

# Plot regret
for poly_degree, regret in regret_df.iteritems():
    axs[1].plot(regret.index, regret.values, label='%s Degree' % poly_degree)
axs[1].set_ylim(bottom=0.0)
axs[1].grid(True)
axs[1].plot(x_train, [0] * len(x_train), "|", label='train')
axs[1].set_xlim(-exp_params.x_test_max, exp_params.x_test_max)
axs[1].set_xlabel('t value')
axs[1].set_ylabel(r'$\Gamma$')
plt.subplots_adjust(wspace=0, hspace=0.15)
fig.savefig(os.path.join(output_dir, 'least_squares_with_poly_degree.jpg'),
            dpi=200, bbox_inches='tight')

# exit()
# ----------------------------------------------------------- #
# Lambda Experiment

# Create parameters
data_params = DataParameters()
pnml_params = PNMLParameters()
exp_params = ExperimentParameters()
data_params.x_train = [-0.6, 0.2, 0.8]
data_params.y_train = [0.5, 0.3, 0.55]
exp_params.poly_degree_list = [1, 2, 3]
exp_params.lambda_list = [0.0, 0.1, 1.0]
pnml_params.y_max = 200
pnml_params.y_min = -200
pnml_params.dy = 0.001
exp_params.x_test_max = 1.4
exp_params.dx_test = 0.05
exp_params.pnml_type = 'only_theta'
print(data_params)
print(pnml_params)
print(exp_params)

# Execute Lambda
exp_h = Experiment(exp_params, data_params, pnml_params)
exp_h.execute_lambda_search()
regret_df = exp_h.get_regret_df()
exp_df_dict = exp_h.get_exp_df_dict()
x_train, y_train = exp_h.get_train()

# Get argmax predictor
argmax_prediction_dict = get_argmax_prediction(exp_df_dict)

#################
# Plots
plt.style.use('seaborn-dark')
size = plt.figure().get_size_inches()
plt.figure(figsize=(size[0], size[1] * 2))

# Plot argmax prediction
fig, axs = plt.subplots(2, 1)
for lamb, prediction in argmax_prediction_dict.items():
    axs[0].plot(exp_h.x_test_array, prediction, label=r'$\lambda$ = %s' % lamb)
axs[0].set_ylabel('Label')
axs[0].grid(True)
axs[0].plot(x_train, y_train, 'o', label='train')
axs[0].set_xlim(-exp_params.x_test_max, exp_params.x_test_max)
axs[0].set_xlabel('')
axs[0].legend()
plt.setp(axs[0].get_xticklabels(), visible=False)

# Plot regret
for lamb, regret in regret_df.iteritems():
    axs[1].plot(regret.index, regret.values, label=r'$\lambda$ = %s' % lamb)
axs[1].set_ylim(bottom=0.0)
axs[1].grid(True)
axs[1].plot(x_train, [0] * len(x_train), '|', label='train')
axs[1].set_xlim(-exp_params.x_test_max, exp_params.x_test_max)
axs[1].set_xlabel('t value')
axs[1].set_ylabel(r'$\Gamma$')
plt.subplots_adjust(wspace=0, hspace=0.1)
fig.savefig(os.path.join(output_dir, 'least_squares_with_regularization.jpg'),
            dpi=200, bbox_inches='tight')
