import os

import matplotlib.pyplot as plt

from data_utilities import DataParameters
from experimnet_utilities import Experiment, ExperimentParameters
from pnml_utilities import PNMLParameters
from general_utilies import autoscale_y

output_dir = '../output/figures'
save_prefix = 'vanilla_'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create parameters
data_params = DataParameters()
pnml_params = PNMLParameters()
exp_params = ExperimentParameters()
exp_params.poly_degree_list = [2, 3]
exp_params.pnml_type = 'no_sigma'
print(data_params)
print(pnml_params)
print(exp_params)
save_prefix = exp_params.pnml_type + '_'

exp_h = Experiment(exp_params, data_params, pnml_params)
exp_h.execute()
regret_df = exp_h.get_regret_df()
exp_df_dict = exp_h.get_exp_df_dict()
x_train, y_train = exp_h.get_train()

# Get best predictor
argmax_prediction_dict = {}
for poly_degree, exp_df in exp_df_dict.items():
    # exp_df: columns: x test. iloc: pdf on y
    argmax_prediction_dict[poly_degree] = exp_df.idxmax(axis=0).tolist()

#################
# Plots


# Plot argmax
fig, axs = plt.subplots(2, 1)  # , constrained_layout=True)
for ax in axs:
    for poly_degree, argmax_prediction in argmax_prediction_dict.items():
        ax.plot(exp_h.x_test_array, argmax_prediction, label='%s Degree' % poly_degree)
    ax.set_ylabel('y value')
    ax.grid(True)
    ax.plot(x_train, y_train, 'o', label='train')
axs[1].set_xlim(-1.2, 1.2)
autoscale_y(axs[1])
axs[1].set_xlabel('x value')
axs[0].set_title(r"argmax$_{y_N}$ $Q(y_N|z^N,x_N)$", fontsize=16)
axs[0].legend()
plt.savefig(os.path.join(output_dir, save_prefix + 'pnml_argmax_prediction.jpg'), dpi=200, bbox_inches='tight')
plt.show()

# Plot regret
fig, axs = plt.subplots(2, 1)  # , constrained_layout=True)
for ax in axs:
    for poly_degree, regret in regret_df.iteritems():
        ax.plot(regret.index, regret.values, label='%s Degree' % poly_degree)
    ax.set_ylabel('Regret')
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.plot(x_train, [0] * len(x_train), 'o', label='train')
axs[1].set_xlim(-1.2, 1.2)
autoscale_y(axs[1])
axs[1].set_xlabel('x value')
axs[0].set_title('pNML Regret', fontsize=16)
axs[0].legend()
plt.savefig(os.path.join(output_dir, save_prefix + 'pnml_regret.jpg'), dpi=200, bbox_inches='tight')
plt.show()

# Plot Distributions
x_val_to_plot_list = [-1.20, 0.70, 4.00]
x_lim_diff_list = [15, 7.5, 200]
fig, axs = plt.subplots(len(x_val_to_plot_list), 1)  # , constrained_layout=True)
for ax, x_val_to_plot, x_lim_diff in zip(axs, x_val_to_plot_list, x_lim_diff_list):
    min_poly_degree = 100
    for poly_degree, exp_df in exp_df_dict.items():
        ax.plot(exp_df.index, exp_df[x_val_to_plot].values, label='%s Degree' % poly_degree)
    ax.set_xlabel('y value distribution at x={:.2f}'.format(x_val_to_plot))
    ax.set_xlim(- x_lim_diff, x_lim_diff)
    ax.set_ylabel('PDF')
    ax.grid(True)
axs[0].legend()
axs[0].set_title(r'pNML $Q(y_N|z^N,x_N)$ Distribution')
fig.subplots_adjust(hspace=0.75)
plt.savefig(os.path.join(output_dir, save_prefix + 'pnml_distribution.jpg'), dpi=200, bbox_inches='tight')
plt.show()

