#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import get_case_config, get_case_sensitivities, STATISTIC_COLNAME
import sensitivity, utils
from strategies import RETURN_KEYS


pinch_point_params = [
        'testing_delay',
        'manual_trace_time',
        'app_cov',
        'trace_adherence',
    ]

sensitivity_params = [
            'latent_period',  # just put this in with pinch points for now...
            'inf_profile',
            'infection_proportions',
            'p_day_noticed_symptoms'
        ]


def nice_lockdown_name(name):
    return name.replace("_", " ").title().replace("Tti", "TTI")


def take_key(res_list, key):
    means = np.array([res[key].iloc[0] for res in res_list])
    standard_errors = np.array([res[key].iloc[1] for res in res_list])
    return means, standard_errors


def nice_param_name(name):
    if name.lower() == 'app_cov':
        return "Application Uptake"
    if name.lower() == 'trace_adherence':
        return "Compliance"
    return name.replace("_", " ").title()


def rand_jitter(arr):
    stdev = .02*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def errorbar(ax, xaxis, means, stds, label):
    conf_intervals = 1.96 * stds
    ax.errorbar(rand_jitter(xaxis), means, yerr=conf_intervals, fmt='o', label=label, lw=1)


def plot_sim_results(ax, sim_results, key, label):
    xvals, reslist = zip(*sim_results)
    reslist = [k.set_index(STATISTIC_COLNAME, drop=True) for k in reslist]
    arg_order = np.argsort(xvals)
    xaxis = np.array(xvals)[arg_order]
    means, standard_errors = take_key(reslist, key)
    values = means[arg_order]
    return errorbar(ax, xaxis, values, standard_errors[arg_order], label)


def legend(fig, ax, ncol=4):
    return fig.legend(
        *ax.get_legend_handles_labels(),
        ncol=ncol,
        bbox_to_anchor=(0.45, -0.01),
        loc="lower center",
        fancybox=False,
    )


def plot_lockdown(lockdown_dct, deck, keys_to_plot):
    for param_name, sim_results in lockdown_dct.items():
        fig, axarr = plt.subplots(1, len(keys_to_plot), sharex=True, squeeze=False)
        for key, ax in zip(keys_to_plot, axarr.flat):
            for lockdown_name, res in sim_results.items():
                plot_sim_results(ax, res, key, nice_lockdown_name(lockdown_name))
            
            ax.set_ylabel(key)
            ax.set_xlabel(nice_param_name(param_name))
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        legend(fig, ax)
        # fig.suptitle(nice_param_name(param_name), y=0.95)
        plt.subplots_adjust(wspace=0.05)
        
        deck.add_figure(fig, name=param_name)
    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    import utils

    # data_dir = os.path.join(, "pinch-points")
    # lockdowns = next(os.walk(data_dir))[1]

    # keys_to_plot = [RETURN_KEYS.reduced_r, RETURN_KEYS.tests]
    # rc_dct = {
        # 'figure.figsize': (14, 6),
        # 'figure.max_open_warning': 1000,
    # }
    # output_folder = os.path.join(os.environ['REPOS'], 'tti-explorer', 'charts')

    # pinch_points_results = defaultdict(lambda: defaultdict(list))

    # for lockdown in lockdowns:
        # folder = os.path.join(data_dir, lockdown)
        # for cfg_file in filter(lambda x: x.startswith("config") and x.endswith('.json'), os.listdir(folder)):
            # i = int(cfg_file.replace("config_", '').replace(".json", ''))
            # cfg = utils.read_json(os.path.join(folder, cfg_file))
            # target = cfg[sensitivity.TARGET_KEY]
            # results = pd.read_csv(os.path.join(folder, f"run_{i}.csv"), index_col=0)
            # pinch_points_results[lockdown][target].append((cfg['config'][target], results))

    # # group by lockdown level and then again by parameter
    # lockdown_results = dict()
    # for i in range(1, 6):
        # lockdown_results[f"S{i}"] = {k: v for k, v in pinch_points_results.items() if int(k[1]) == i}
    # lockdown_results = {k: utils.swaplevel(v) for k,v in lockdown_results.items()}

    # with plt.rc_context(rc_dct):
        # for level, results in lockdown_results.items():
            # deck = utils.PdfDeck()
            # plot_lockdown(results, deck, keys_to_plot)
            # deck.make(os.path.join(output_folder, f"{level}_pinch_points.pdf"))
            # deck.make_individual(folder=os.path.join(f"{level}_individual", output_folder))

