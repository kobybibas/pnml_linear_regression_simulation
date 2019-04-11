import matplotlib.pyplot as plt

from general_utilies import autoscale_y


def plot_prediction(prediction_dict, x_train, y_train, x_test, fig=None, axs=None):
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 1)
    for ax in axs:
        for poly_degree, prediction in prediction_dict.items():
            ax.plot(x_test, prediction, label='%s Degree' % poly_degree)
        ax.set_ylabel('y value')
        ax.grid(True)
        ax.plot(x_train, y_train, 'o', label='train')
    axs[1].set_xlim(-0.6, 0.6)
    autoscale_y(axs[1])
    axs[1].set_xlabel('x value')
    axs[0].set_title(r"argmax$_{y_N}$ $Q(y_N|z^N,x_N)$", fontsize=16)
    axs[0].legend()
    return fig, axs


def plot_distributions_poly_degree(exp_df_dict, x_val_to_plot_list, x_lim_diff_list, fig=None, axs=None):
    # Plot Distributions
    if fig is None or axs is None:
        fig, axs = plt.subplots(len(x_val_to_plot_list), 1)  # , constrained_layout=True)
    for ax, x_val_to_plot, x_lim_diff in zip(axs, x_val_to_plot_list, x_lim_diff_list):
        for poly_degree, exp_df in exp_df_dict.items():
            ax.plot(exp_df.index, exp_df[x_val_to_plot].values, label='%s Degree' % poly_degree)
        ax.set_xlabel('y value distribution at x={:.2f}'.format(x_val_to_plot))
        ax.set_xlim(- x_lim_diff, x_lim_diff)
        ax.set_ylabel('PDF')
        ax.grid(True)
    axs[0].legend()
    axs[0].set_title(r'pNML $Q(y_N|z^N,x_N)$ Distribution')
    fig.subplots_adjust(hspace=0.75)
    return fig, axs


def plot_regret_poly_degree(regret_df, x_train, fig=None, axs=None):
    # Plot regret
    if fig is None or axs is None:
        fig, axs = plt.subplots(len(regret_df.columns), 1)
    for ax in axs:
        for poly_degree, regret in regret_df.iteritems():
            ax.plot(regret.index, regret.values, label='%s Degree' % poly_degree)
        ax.set_ylabel('Regret')
        ax.set_ylim(bottom=0.0)
        ax.grid(True)
        ax.plot(x_train, [0] * len(x_train), 'o', label='train')
    axs[1].set_xlim(-0.6, 0.6)
    autoscale_y(axs[1])
    axs[1].set_xlabel('x value')
    axs[0].set_title('pNML Regret', fontsize=16)
    axs[0].legend()
    return fig, axs
