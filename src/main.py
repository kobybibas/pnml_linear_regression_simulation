import os

from data_utilities import DataParameters
from experimnet_utilities import Experiment, ExperimentParameters
from plot_utilities import plot_distributions_poly_degree, plot_regret_poly_degree, plot_prediction
from pnml_utilities import PNMLParameters
from pnml_utilities import twice_universality, get_argmax_prediction, get_mean_prediction

output_dir = '../output/figures'
save_prefix = 'vanilla_'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create parameters
data_params = DataParameters()
pnml_params = PNMLParameters()
exp_params = ExperimentParameters()
exp_params.poly_degree_list = [1, 2]
pnml_params.y_max = 20000
pnml_params.y_min = -100
exp_params.x_test_max = 2.6
data_params.x_train = [-0.5, 0.2, 0.3]
data_params.y_train = [0.5, 0.3, 0.55]
exp_params.pnml_type = 'only_theta'
print(data_params)
print(pnml_params)
print(exp_params)
save_prefix = 'both' + '_'

for pnml_type in ['joint', 'only_theta']:
    exp_params.pnml_type = pnml_type
    exp_h = Experiment(exp_params, data_params, pnml_params)
    exp_h.execute_poly_degree_search()
    regret_df = exp_h.get_regret_df()
    exp_df_dict = exp_h.get_exp_df_dict()
    x_train, y_train = exp_h.get_train()

    # Twice universal
    twice_df = twice_universality(exp_df_dict)
    exp_df_dict['Twice'] = twice_df

    # Get argmax predictor
    print('argmax predictor')
    argmax_prediction_dict = get_argmax_prediction(exp_df_dict)
    for key in argmax_prediction_dict:
        argmax_prediction_dict[pnml_type + '_' + key] = argmax_prediction_dict.pop(key)

    # Get mean predictor
    print('mean predictor')
    mean_prediction_dict = get_mean_prediction(exp_df_dict)
    for key in mean_prediction_dict:
        mean_prediction_dict[pnml_type + '_' + key] = mean_prediction_dict.pop(key)

    #################
    # Plots

    # Plot argmax prediction
    if 'fig_argmax_pred' not in locals() or 'axs_argmax_pred' not in locals():
        fig_argmax_pred, axs_argmax_pred = None, None
    fig_argmax_pred, axs_argmax_pred = plot_prediction(argmax_prediction_dict, x_train, y_train, exp_h.x_test_array,
                                                       fig_argmax_pred, axs_argmax_pred)
    fig_argmax_pred.savefig(os.path.join(output_dir, save_prefix + 'pnml_argmax_prediction.jpg'),
                            dpi=200, bbox_inches='tight')
    # plt.show()

    # Plot mean prediction
    if 'fig_mean_pred' not in locals() or 'axs_mean_pred' not in locals():
        fig_mean_pred, axs_mean_pred = None, None
    fig_mean_pred, axs_mean_pred = plot_prediction(mean_prediction_dict, x_train, y_train, exp_h.x_test_array,
                                                   fig_mean_pred, axs_mean_pred)
    axs_mean_pred[0].set_title(r"mean$_{y_N}$ $Q(y_N|z^N,x_N)$", fontsize=16)
    fig_mean_pred.savefig(os.path.join(output_dir, save_prefix + 'pnml_mean_prediction.jpg'), dpi=200,
                          bbox_inches='tight')
    # plt.show()

    # Plot regret
    if 'fig_regret' not in locals() or 'axs_regret' not in locals():
        fig_regret, axs_regret = None, None
    fig_regret, axs_regret = plot_regret_poly_degree(regret_df, x_train,
                                                     fig_regret, axs_regret)
    fig_regret.savefig(os.path.join(output_dir, save_prefix + 'pnml_regret.jpg'), dpi=200, bbox_inches='tight')
    # plt.show()

    # Plot Distributions
    if 'fig_dist' not in locals() or 'axs_dist' not in locals():
        fig_dist, axs_dist = None, None
    x_val_to_plot_list = [-1.20, 0.20, exp_h.x_test_array[-1]]
    x_lim_diff_list = [15, 7.5, pnml_params.y_max]
    fig_dist, axs_dist = plot_distributions_poly_degree(exp_df_dict, x_val_to_plot_list, x_lim_diff_list,
                                                        fig_dist, axs_dist)
    fig_dist.savefig(os.path.join(output_dir, save_prefix + 'pnml_distribution.jpg'), dpi=200, bbox_inches='tight')
    # plt.show()

pass
