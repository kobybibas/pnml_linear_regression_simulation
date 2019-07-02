import os

import pandas as pd


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot, top)


def exp_df_dict_saver(dictex, output_dir):
    for key, val in dictex.items():
        val.to_csv(os.path.join(output_dir, "exp_df_dict_{}.csv".format(str(key))))

    with open(os.path.join(output_dir, "exp_df_dict_keys.txt"), "w") as f:  # saving keys to file
        f.write(str(list(dictex.keys())))


def exp_df_dict_loader(output_dir):
    """Reading data from keys"""
    with open(os.path.join(output_dir, "exp_df_dict_keys.txt"), "r") as f:
        keys = eval(f.read())

    dictex = {}
    for key in keys:
        dictex[key] = pd.read_csv(os.path.join(output_dir, "exp_df_dict_{}.csv".format(str(key))),
                                  index_col=0
                                  )

    return dictex
