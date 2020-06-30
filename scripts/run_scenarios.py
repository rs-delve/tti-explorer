import os
import warnings
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from tti_explorer import config, strategies, utils
from tti_explorer.scenario import get_monte_carlo_factors, run_scenario, results_table, scale_results


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("strategy", help="The name of the strategy to use", type=str)
    parser.add_argument(
        "population",
        help=(
            "Folder containing population files, "
            "we will assume all .json files in folder are to be  used."
        ),
        type=str,
    )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if not specified.",
        type=str,
    )
    parser.add_argument(
        "--scenarios",
        help=(
            "Which scenarios to run from config.py."
            "By default, if no value is given, all scenarios available for a given strategy are run."
        ),
        nargs="*",
    )
    parser.add_argument(
        "--seed",
        help="Seed for random number generator. All runs will be re-seeded with this value. Default %(default)s.",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    strategy = strategies.registry[args.strategy]
    strategy_configs = config.get_strategy_configs(
        args.strategy, args.scenarios  # this will be None if not specified
    )

    scenario_results = defaultdict(dict)
    case_files = utils.find_case_files(args.population)
    pbar = tqdm(
        desc="Running strategies",
        total=len(case_files) * len(strategy_configs),
        smoothing=0,
    )

    case_metadata = dict()
    for i, case_file in enumerate(case_files):
        case_contacts, metadata = utils.load_cases(
            os.path.join(args.population, case_file)
        )
        case_file_key = utils.tidy_fname(case_file)
        case_metadata[case_file_key] = metadata

        _, symp_covid_pos, asymp_covid_pos = metadata["case_config"]["infection_proportions"]["dist"]
        monte_carlo_factor, r_monte_carlo_factor = get_monte_carlo_factors(len(case_contacts), symp_covid_pos, asymp_covid_pos)

        rng = np.random.RandomState(seed=args.seed)
        nppl = metadata["case_config"]["infection_proportions"]["nppl"]
        for scenario, cfg_dct in strategy_configs.items():
            r = run_scenario(case_contacts, strategy, rng, cfg_dct)
            scenario_results[scenario][case_file_key] = scale_results(
                r, monte_carlo_factor, r_monte_carlo_factor, nppl
            )

            pbar.update(1)

    # dct = {k: pd.concat(v, axis=1).T for k, v in scenario_results.items()}
    # df = pd.concat(dct, axis=0)

    tables = dict()
    os.makedirs(args.output_folder, exist_ok=True)
    for scenario, v in scenario_results.items():
        table = results_table(v, "case_file")
        table.to_csv(os.path.join(args.output_folder, f"{scenario}.csv"))
        tables[scenario] = table
    utils.write_json(
        case_metadata, os.path.join(args.output_folder, "case_metadata.json")
    )
    # df = pd.DataFrame.from_dict(tables, orient="index")
    # df.groupby(level=0).agg(['mean', 'std']).to_csv(os.path.join(args.output_folder, 'all_results.csv'))

    all_results = pd.concat(tables)
    all_results.index.set_names("scenario", level=0, inplace=True)

    all_results.to_csv(os.path.join(args.output_folder, "all_results.csv"))
    all_results.unstack(level=-1).to_csv(
        os.path.join(args.output_folder, "all_results_pivot.csv")
    )
