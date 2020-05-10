import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contacts import Contacts, NCOLS
from generate_cases import Case

from utils import named_product

def find_case_file(folder, fname):
    return next(filter(lambda x: x == fname, os.listdir(folder)))

def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)

def load_results(fpath):
    # only return reduced_r, manual_traces, tests_needed and persondays_quarantined
    results = pd.read_csv(fpath, usecols = ['Reduced R', 'Manual Traces', 'Tests Needed'])
    if results.ndim > 1:
        results = results.mean(axis = 0)
    return results

def max_calculator(folder, tti_strat_list, gov_measure_list):
    curr_max = np.zeros(3)
    for gov_measure in gov_measure_list:
        for tti_strat in tti_strat_list:
            tti_fname = gov_measure + tti_strat
            tti_file = find_case_file(folder, tti_fname)
            tti_results = load_results(os.path.join(folder, tti_file))
            curr_max = np.maximum(curr_max, tti_results)
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
            "we will assume all results files are named L{x}_{tti_measure}.csv for 1 <= x <= 5."),
        type=str
        )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if not specified.",
        type=str
        )

    args = parser.parse_args()

    tti_strat_list = ['_symptom_tracing.csv', '_positive_test_tracing.csv', '_positive_test_tracing_test_contacts.csv']
    tti_strat_formal_list = ['Trace on symptoms', 'Trace on positive test', 'Test on positive test']
    tti_strat_combined_list = list(zip(tti_strat_list, tti_strat_formal_list))

    metric_list = ['Reduced R', 'Manual Traces', 'Tests Needed']
    metric_formal_list = ['Reduced R', 'Manual Traces (K)', 'Tests Needed (K)']
    metric_combined_list = list(zip(metric_list, metric_formal_list))

    gov_measures = ['L5', 'L4', 'L3', 'L2', 'L1']

    max = max_calculator(args.results_folder, tti_strat_list + ['_no_contact_tracing.csv'], gov_measures)
    ylim_list = list(zip(np.zeros(3), max))

    plt_list = named_product(row = np.arange(3), col = np.arange(3))
    fig, axs = plt.subplots(3, 3, figsize = (12, 12))

    for plt_idx, (row_idx, col_idx) in enumerate(plt_list):
        ax = axs[row_idx, col_idx]
        metric, metric_formal = metric_combined_list[row_idx]
        tti_strat, tti_strat_formal = tti_strat_combined_list[col_idx]

        no_tti = []
        tti = []
        for L_idx, gov_measure in enumerate(gov_measures):
            tti_fname = gov_measure + tti_strat
            tti_file = find_case_file(args.results_folder, tti_fname)
            tti_results = load_results(os.path.join(args.results_folder, tti_file))
            tti.append(tti_results[metric])

            no_tti_fname = gov_measure + '_no_contact_tracing.csv'
            no_tti_file = find_case_file(args.results_folder, no_tti_fname)
            no_tti_results = load_results(os.path.join(args.results_folder, no_tti_file))
            no_tti.append(no_tti_results[metric])

        # sort y axis
        ax.set(ylabel =  metric_formal)
        ax.set_ylim(ylim_list[row_idx])

        # sort x axis
        xlabels = np.arange(5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(gov_measures)

        ax.plot(xlabels, no_tti, alpha = 0.7, marker = 'x', label = 'No TTI')
        ax.plot(xlabels, tti, alpha = 0.7, marker = 'x', label = f'{tti_strat_formal}', color = f'C{col_idx + 1}')

        if row_idx == 0:
            ax.set_title(tti_strat_formal, fontsize = 10)
        ax.set_xlabel('Levels of physical distancing measures')

        ax.legend(loc = 2)
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(args.output_folder, 'gov_measures.pdf'))
