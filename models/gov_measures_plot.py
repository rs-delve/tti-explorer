import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contacts import Contacts, NCOLS
from generate_cases import Case

from utils import named_product

def run_scenario(case_contacts, strategy, rng, strategy_cgf_dct):
    return pd.DataFrame([strategy(*cc, rng, **strategy_cgf_dct) for cc in case_contacts])


def find_case_file(folder, start):
    return next(filter(lambda x: x.startswith(start), os.listdir(folder)))

def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)

def load_results(fpath, scale = 1000):
    # only return reduced_r, manual_traces, tests_needed and persondays_quarantined
    results =  np.genfromtxt(fpath, delimiter=',', skip_header = 1, usecols = (2, 3, 5))
    mean_result = np.mean(results, axis = 0)
    scales = np.array([1., scale, scale])
    mean_result /= scales
    # average over populations
    return np.mean(results, axis = 0)

def max_calculator(folder, tti_strat_list, gov_measure_list):
    curr_max = np.zeros(3)
    for gov_measure in gov_measure_list:
        for tti_strat in tti_strat_list:
            tti_fname = gov_measure + tti_strat
            tti_file = find_case_file(folder, tti_fname)
            tti_results = load_results(os.path.join(folder, tti_file))
            curr_max = np.maximum(curr_max, tti_results)
    return curr_max

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
        "population",
        help=("Folder containing population files, "
            "we will assume all .json files in folder are to be  used and begin with L."),
        type=str
        )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if not specified.",
        type=str
        )

    args = parser.parse_args()

    tti_strat_list = ['_symptom_tracing', '_positive_test_tracing', '_positive_test_tracing_test_contacts']
    tti_strat_formal_list = ['Trace on symptoms', 'Trace on positive test', 'Test on positive test']

    metric_list = ['Reduced R', 'Manual Traces (K)', 'Tests Needed (K)']
    gov_measures = ['L5', 'L4', 'L3', 'L2', 'L1']

    max = max_calculator(args.population, tti_strat_list + ['_no_contact_tracing'], gov_measures)
    ylim_list = list(zip(np.zeros(3), max))

    plt_list = named_product(row = np.arange(3), col = np.arange(3))
    fig, axs = plt.subplots(3, 3, figsize = (12, 12))

    for plt_idx, (row_idx, col_idx) in enumerate(plt_list):
        ax = axs[row_idx, col_idx]
        no_tti = []
        tti = []
        for L_idx, gov_measure in enumerate(gov_measures):
            tti_fname = gov_measure + tti_strat_list[col_idx]
            tti_file = find_case_file(args.population, tti_fname)
            tti_results = load_results(os.path.join(args.population, tti_file))
            tti.append(tti_results[row_idx])

            no_tti_fname = gov_measure + '_no_contact_tracing'
            no_tti_file = find_case_file(args.population, no_tti_fname)
            no_tti_results = load_results(os.path.join(args.population, no_tti_file))
            no_tti.append(no_tti_results[row_idx])

        # sort y axis
        ax.set(ylabel =  metric_list[row_idx])
        # ax.set_yticks(metric_yticks_list[row_idx])
        ax.set_ylim(ylim_list[row_idx])

        # sort x axis
        xlabels = np.arange(5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(gov_measures)

        ax.plot(xlabels, tti, alpha = 0.7, marker = 'x', label = 'New TTI policy')
        ax.plot(xlabels, no_tti, alpha = 0.7, marker = 'x', label = 'No TTI policy')

        if row_idx == 0:
            ax.set_title(tti_strat_formal_list[col_idx], fontsize = 10)
            if col_idx == 0:
                ax.legend()

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(args.output_folder, 'gov_measures.pdf'))
