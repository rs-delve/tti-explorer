import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-ticks')

from contacts import Contacts, NCOLS
from generate_cases import Case

from strategies import RETURN_KEYS
from utils import named_product


def find_case_file(folder, fname):
    return next(filter(lambda x: x == fname, os.listdir(folder)))

def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)

def load_results(fpath):
    # only return reduced_r, manual_traces, tests_needed
    results = pd.read_csv(fpath, index_col=[0], usecols=['statistic', RETURN_KEYS.reduced_r, RETURN_KEYS.man_trace, RETURN_KEYS.tests, RETURN_KEYS.quarantine])
    # results = results.head(2)
    if len(results) > 2:
        raise ValueError(f"More than 1 population found in {fpath}")
    return results

def max_calculator(folder, tti_strat_list, gov_measure_list):
    curr_max = np.zeros(len(tti_strat_list))
    for gov_measure in gov_measure_list:
        for tti_strat in tti_strat_list:
            tti_fname = gov_measure + tti_strat
            tti_file = find_case_file(folder, tti_fname)
            tti_results = load_results(os.path.join(folder, tti_file))
            curr_max = np.maximum(curr_max, tti_results.loc['mean'].values)
    return curr_max * 1.2

if __name__ == "__main__":
    from argparse import ArgumentParser
    from collections import defaultdict
    from datetime import datetime
    import time
    from types import SimpleNamespace
    import os

    from tqdm import tqdm

    import config
    from strategies import registry

    parser = ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument(
        "results_folder",
        help=("Folder containing results files, "
            "we will assume all results files are named S{x}_{tti_measure}.csv for 1 <= x <= 5."),
        type=str
        )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if not specified.",
        type=str
        )

    args = parser.parse_args()

    tti_strat_list = ['_symptom_based_TTI.csv', '_test_based_TTI.csv', '_test_based_TTI_test_contacts.csv']
    tti_strat_formal_list = ['Symptom-based TTI', 'Test-based TTI', 'Test-based TTI, test contacts']
    tti_strat_combined_list = list(zip(tti_strat_list, tti_strat_formal_list))
    no_tti_str = '_no_TTI.csv'

    metric_list = [RETURN_KEYS.reduced_r, RETURN_KEYS.man_trace, RETURN_KEYS.tests, RETURN_KEYS.quarantine]
    metric_formal_list = [RETURN_KEYS.reduced_r, RETURN_KEYS.man_trace + " (K)", RETURN_KEYS.tests + " (K)", "# Person-days in Quarantine (K)"]
    metric_combined_list = list(zip(metric_list, metric_formal_list))

    gov_measures = ['S5', 'S4', 'S3', 'S2', 'S1']

    max = max_calculator(args.results_folder, tti_strat_list + [no_tti_str], gov_measures)
    ylim_list = list(zip(np.zeros(len(metric_list)), max))

    plt_list = named_product(row = np.arange(len(metric_list)), col = np.arange(len(tti_strat_list)))
    fig, axs = plt.subplots(len(metric_list), len(tti_strat_list), figsize = (12, 12))

    for plt_idx, (row_idx, col_idx) in enumerate(plt_list):
        ax = axs[row_idx, col_idx]
        metric, metric_formal = metric_combined_list[row_idx]
        tti_strat, tti_strat_formal = tti_strat_combined_list[col_idx]

        no_tti = []
        tti = []

        no_tti_std_error = []
        tti_std_error = []

        for gov_measure in gov_measures:
            tti_fname = gov_measure + tti_strat
            tti_file = find_case_file(args.results_folder, tti_fname)
            tti_results = load_results(os.path.join(args.results_folder, tti_file))
            tti.append(tti_results[metric].loc['mean'])
            tti_std_error.append(tti_results[metric].loc['std'])

            no_tti_fname = gov_measure + no_tti_str
            no_tti_file = find_case_file(args.results_folder, no_tti_fname)
            no_tti_results = load_results(os.path.join(args.results_folder, no_tti_file))
            no_tti.append(no_tti_results[metric].loc['mean'])
            no_tti_std_error.append(no_tti_results[metric].loc['std'])

        # sort y axis
        if col_idx == 0:
            ax.set(ylabel =  metric_formal)
        ax.set_ylim(ylim_list[row_idx])

        # sort x axis
        xlabels = np.arange(5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(gov_measures)

        if metric == RETURN_KEYS.reduced_r:
            ax.set_title(tti_strat_formal, fontsize = 10)
            ax.hlines(1, 0, 4, 'k', ls = '--', alpha = 0.5)

        if metric in (RETURN_KEYS.reduced_r, RETURN_KEYS.quarantine):
            ax.errorbar(
                x = xlabels,
                y = no_tti,
                yerr = 1.96 * np.array(no_tti_std_error),
                #ls = 'None',
                label = 'No TTI',
                # marker = '.',
                capsize=2,
                markersize = 10
            )

        if metric == metric_list[-1]:
            ax.set_xlabel('Level of NPIs')

        #ax.plot(xlabels, tti, color=f'C{col_idx + 1}')
        ax.errorbar(
            x = xlabels,
            y = tti, yerr = 1.96 * np.array(tti_std_error),
            #ls = 'None',
            label = tti_strat_formal,
            color = f'C{col_idx + 1}',
            # marker = '.',
            capsize=2,
            markersize = 10
        )

        ax.grid(False)
        # ax.plot(xlabels, no_tti, alpha = 0.7, marker = 'x', label = 'No TTI')
        # ax.plot(xlabels, tti, alpha = 0.7, marker = 'x', label = f'{tti_strat_formal}', color = f'C{col_idx + 1}')
        #
        # ax.fill_between(
        #     xlabels,
        #     np.array(no_tti) + (1.96*np.array(no_tti_std_error)),
        #     np.array(no_tti) - (1.96*np.array(no_tti_std_error)),
        #     alpha=0.5
        # )
        # ax.fill_between(
        #     xlabels,
        #     np.array(tti) + (1.96*np.array(tti_std_error)),
        #     np.array(tti) - (1.96*np.array(tti_std_error)),
        #     alpha=0.5,
        #     color = f'C{col_idx + 1}'
        # )


        ax.legend(loc = 2)
    # for ax in axs.flat:
    #     ax.label_outer()

    plt.savefig(os.path.join(args.output_folder, 'gov_measures.pdf'))
