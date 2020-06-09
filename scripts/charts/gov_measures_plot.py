import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tti_explorer.strategies import RETURN_KEYS
from tti_explorer.utils import named_product
import tti_explorer.config as config


def find_results_file(folder, fname):
    return next(filter(lambda x: x == fname, os.listdir(folder)))


def load_results(fpath):
    results = pd.read_csv(
        fpath,
        index_col=[0],
        usecols=[
            config.STATISTIC_COLNAME,
            RETURN_KEYS.reduced_r,
            RETURN_KEYS.man_trace,
            RETURN_KEYS.tests,
            RETURN_KEYS.quarantine,
        ],
    )

    # We expect two rows: mean and standard error
    if len(results) > 2:
        raise ValueError(f"More than 1 population found in {fpath}")

    return results


def max_calculator(folder, tti_strat_list, gov_measure_list):
    curr_max = np.zeros(len(tti_strat_list))
    for gov_measure in gov_measure_list:
        for tti_strat in tti_strat_list:
            tti_fname = gov_measure + tti_strat
            tti_file = find_results_file(folder, tti_fname)
            tti_results = load_results(os.path.join(folder, tti_file))
            curr_max = np.maximum(curr_max, tti_results.loc["mean"].values)
    return curr_max * 1.2


if __name__ == "__main__":  # noqa: C901
    from argparse import ArgumentParser
    import os

    plt.style.use("seaborn-ticks")

    parser = ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument(
        "results_folder",
        help=(
            "Folder containing results files, "
            "we will assume all results files are named S{x}_{tti_measure}.csv for 1 <= x <= 5."
        ),
        type=str,
    )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if not specified.",
        type=str,
    )

    args = parser.parse_args()

    tti_strat_list = [
        "_no_TTI.csv",
        "_symptom_based_TTI.csv",
        "_test_based_TTI.csv",
        "_test_based_TTI_test_contacts.csv",
    ]
    tti_strat_formal_list = [
        "No TTI",
        "Symptom-based TTI",
        "Test-based TTI",
        "Test-based TTI, test contacts",
    ]
    tti_strat_combined_list = list(zip(tti_strat_list, tti_strat_formal_list))

    test_based_combined_formal_str = "Both Test-based TTIs"
    test_based_combined_idx = 2
    not_test_based_test_contacts_formal_str = "Symptom-based TTI & Test-based TTI"
    not_test_based_test_contacts_idx = 1

    metric_list = [
        RETURN_KEYS.reduced_r,
        RETURN_KEYS.man_trace,
        RETURN_KEYS.tests,
        RETURN_KEYS.quarantine,
    ]
    metric_formal_list = [
        RETURN_KEYS.reduced_r,
        RETURN_KEYS.man_trace,
        RETURN_KEYS.tests,
        "# Person-days in Quarantine",
    ]
    metric_ylabel_list = [RETURN_KEYS.reduced_r] + ["Thousands"] * (
        len(metric_list) - 1
    )
    metric_combined_list = list(
        zip(metric_list, metric_formal_list, metric_ylabel_list)
    )

    gov_measures = ["S5", "S4", "S3", "S2", "S1"]

    x_axis_jitter = [-0.01, -0.005, 0.005, 0.01]

    max = max_calculator(args.results_folder, tti_strat_list, gov_measures)
    ylim_list = list(zip(np.zeros(len(metric_list)), max))

    plt_size = int(np.sqrt(len(metric_list)))
    plt_list = named_product(row=np.arange(plt_size), col=np.arange(plt_size))
    fig, axs = plt.subplots(plt_size, plt_size, figsize=(12, 12))

    for plt_idx, (row_idx, col_idx) in enumerate(plt_list):
        ax = axs[row_idx, col_idx]
        metric, metric_formal, metric_ylabel = metric_combined_list[plt_idx]

        # sort y axis
        ax.set(ylabel=metric_ylabel)
        ax.set_ylim(ylim_list[plt_idx])

        # sort x axis
        xlabels = np.arange(5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(gov_measures)
        ax.set_xlabel("Stringency level of NPIs")

        ax.set_title(metric_formal, fontsize=15)

        if metric == RETURN_KEYS.reduced_r:
            ax.hlines(1, 0, 4, "k", ls="--", alpha=0.5)

        for tti_strat_idx, (tti_strat, tti_strat_formal) in enumerate(
            tti_strat_combined_list
        ):
            no_tti = []
            tti = []

            no_tti_std_error = []
            tti_std_error = []

            for gov_measure in gov_measures:
                tti_fname = gov_measure + tti_strat
                tti_file = find_results_file(args.results_folder, tti_fname)
                tti_results = load_results(os.path.join(args.results_folder, tti_file))
                tti.append(tti_results[metric].loc["mean"])
                tti_std_error.append(tti_results[metric].loc["std"])

            if metric in (RETURN_KEYS.man_trace,):
                if tti_strat_formal.startswith("Test-based TTI, test"):
                    ax.errorbar(
                        x=xlabels + x_axis_jitter[tti_strat_idx],
                        y=tti,
                        yerr=1.96 * np.array(tti_std_error),
                        label=test_based_combined_formal_str,
                        color=f"C{test_based_combined_idx}",
                        capsize=2,
                        markersize=10,
                        alpha=0.7,
                    )
                elif tti_strat_formal.startswith("Symptom"):
                    ax.errorbar(
                        x=xlabels + x_axis_jitter[tti_strat_idx],
                        y=tti,
                        yerr=1.96 * np.array(tti_std_error),
                        label=tti_strat_formal,
                        color=f"C{tti_strat_idx}",
                        capsize=2,
                        markersize=10,
                        alpha=0.7,
                    )
            elif metric in (RETURN_KEYS.tests,):
                if tti_strat_formal.startswith("Test-based TTI, test"):
                    ax.errorbar(
                        x=xlabels,
                        y=tti,
                        yerr=1.96 * np.array(tti_std_error),
                        label=tti_strat_formal,
                        color=f"C{tti_strat_idx}",
                        capsize=2,
                        markersize=10,
                        alpha=0.7,
                    )
                elif tti_strat_formal == "Test-based TTI":
                    ax.errorbar(
                        x=xlabels,
                        y=tti,
                        yerr=1.96 * np.array(tti_std_error),
                        label=not_test_based_test_contacts_formal_str,
                        color=f"C{not_test_based_test_contacts_idx}",
                        capsize=2,
                        markersize=10,
                        alpha=0.7,
                    )
            else:
                jitter = (
                    np.zeros(4) if metric == RETURN_KEYS.quarantine else x_axis_jitter
                )
                ax.errorbar(
                    x=xlabels + jitter[tti_strat_idx],
                    y=tti,
                    yerr=1.96 * np.array(tti_std_error),
                    label=tti_strat_formal,
                    color=f"C{tti_strat_idx}",
                    capsize=2,
                    markersize=10,
                    alpha=0.7,
                )

        ax.grid(False)

        ax.legend(loc=2)

    plt.savefig(os.path.join(args.output_folder, "gov_measures.pdf"))
