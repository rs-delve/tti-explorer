import json

import numpy as np
import pandas as pd

from contacts import Contacts, NCOLS
from generate_cases import Case


import warnings
warnings.filterwarnings("error")

# def results_table(results_dct, index_name="scenario"):
#     df = pd.DataFrame.from_dict(
#             results_dct,
#             orient="index"
#         ).sort_index()
#     df.index.name = index_name
#     return df


def results_table(results_dct, index_name="scenario"):
    df = {k: v.T for k, v in results_dct.items()}
    df = pd.concat(df)
    # df.index = df.keys()
    df.index.names = [index_name, 'statistic']
    return df


def load_cases(fpath):
    """load_cases
    Loads case and contact from .json file into Cases and Contacts.

    Args:
        fpath (str): path to file.

    Returns (tuple[list[tuple[Case, Contact], dict]):
        pairs: list of Case, Contact pairs
        meta: dictionary of meta-data for case/contact generation
    """
    with open(fpath, "r") as f:
        raw = json.load(f)

    cases = raw.pop("cases")
    meta = raw
    pairs = list()
    for dct in cases:
        case = Case(**dct['case'])

        contacts_dct = dct['contacts']
        n_daily = contacts_dct.pop('n_daily')
        contacts_dct = {k: np.array(v, dtype=int).reshape(-1, NCOLS) for k, v in contacts_dct.items()}
        contacts = Contacts(n_daily=n_daily, **contacts_dct)
        pairs.append((case, contacts))
    return pairs, meta


def run_scenario(case_contacts, strategy, rng, strategy_cgf_dct):
    df = pd.DataFrame([strategy(*cc, rng, **strategy_cgf_dct) for cc in case_contacts])
    return pd.concat({'mean': df.mean(0), 'std': df.std(0)}, axis=1)


def find_case_files(folder, ending=".json"):
    return list(filter(lambda x: x.endswith(ending), os.listdir(folder)))


def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    import json
    import time
    from types import SimpleNamespace
    import os

    from tqdm import tqdm

    import config
    import sensitivity
    import strategies

    from run_scenarios import scale_results
    
    parser = ArgumentParser("Run sensitivity analysis on strategy")
    parser.add_argument(
            "strategy",
            help="The name of the strategy to use",
            type=str
        )
    parser.add_argument(
        "population",
        help=("Folder containing population files, "
            "we will assume all .json files in folder are to be  used."),
        type=str
        )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if needed.",
        type=str
        )
    parser.add_argument(
            "--scenarios",
            help="Which scenarios to run from config.py. If not given then all are run.",
            default=[config.ALL_CFG_FLAG],
            type=str,
            nargs="*"
        )
    parser.add_argument(
            "--sensitivity",
            help=("Method for sensitivity analysis "
                "over parameters designated for sensitivity analysis in config.py. "
                "Empty string does no sensitivity analysis. Default '%(default)s'."),
            default="",
            type=str
        )
    parser.add_argument(
            "--seed",
            help="Seed for random number generator. All runs will be re-seeded with this value. Default %(default)s.",
            default=0,
            type=int
        )
    parser.add_argument(
            "--parameters",
            help="Specific parameters to ablate over. Optional, if not present runs over all parameters defined for a strategy.",
            nargs="*"
        )
    parser.add_argument(
            "--nprocs",
            help="Number of cores on which to run this script. Default %(default)s",
            default=1,
            type=int
        )
    args = parser.parse_args()
    strategy = strategies.registry[args.strategy]
    strategy_configs = config.get_strategy_config(
            args.strategy,
            args.scenarios
        )
    config_generator = sensitivity.registry[args.sensitivity] if args.sensitivity else None

    case_files = find_case_files(args.population)
    pbar = tqdm(
            desc="Running configurations/sensitivities",
            total=len(case_files) * len(strategy_configs) * 20,  # this is just number of entries in temporal anne sensitivities generator
            smoothing=None
        )
    scenario_results = defaultdict(lambda: defaultdict(dict))
    configs_dct = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.nprocs) as executor:
        for case_file in case_files:
            case_contacts, metadata = load_cases(os.path.join(args.population, case_file))
            nppl = metadata['case_config']['infection_proportions']['nppl']
            
            # Can we turn this into something like calculate_confidence_interval?
            n_monte_carlo_samples = len(case_contacts)
            n_r_monte_carlo_samples = len(case_contacts) * (
                    metadata['case_config']['infection_proportions']['dist'][1]
                    + metadata['case_config']['infection_proportions']['dist'][2]
                )
            
            monte_carlo_factor = 1. / np.sqrt(n_monte_carlo_samples)
            r_monte_carlo_factor = 1. / np.sqrt(n_r_monte_carlo_samples)

            for scenario, cfg_dct in strategy_configs.items():
                policy_sensitivities = config.get_policy_sensitivities(args.strategy)
                if args.parameters is not None:
                    policy_sensitivities = dict((k, policy_sensitivities[k]) for k in args.parameters)

                cfgs = config_generator(
                        cfg_dct,
                        policy_sensitivities
                        ) if args.sensitivity else [{sensitivity.CONFIG_KEY: cfg_dct, sensitivity.TARGET_KEY: ""}]
                
                futures = list()
                for i, cfg in enumerate(cfgs):
                    future = executor.submit(
                            run_scenario,
                            case_contacts,
                            strategy,
                            np.random.RandomState(seed=args.seed),
                            cfg[sensitivity.CONFIG_KEY]
                        )
                    futures.append(future)
                    configs_dct[scenario][i] = cfg

                for i, future in enumerate(futures):
                    # this is so uglY!
                    scenario_results[scenario][i][tidy_fname(case_file)] = scale_results(
                        future.result(),
                        monte_carlo_factor,
                        r_monte_carlo_factor,
                        nppl
                    )
                    pbar.update(1)

    os.makedirs(args.output_folder, exist_ok=True)
    for scenario, res_dict in scenario_results.items():
        odir = os.path.join(args.output_folder, scenario)
        os.makedirs(odir, exist_ok=True)
        res_tables = dict()
        for i, res_dct_over_seeds in res_dict.items():
            with open(os.path.join(odir, f"config_{i}.json"), "w") as f:
                    json.dump(
                        dict(
                            configs_dct[scenario][i],
                            seed=args.seed,
                            population=args.population,
                            strategy=args.strategy
                        ),
                        f
                    )

            table = results_table(res_dct_over_seeds)
            table.to_csv(
                os.path.join(
                    odir,
                    f"run_{i}.csv"
                )
            )
            res_tables[i] = table
        over_all_seeds = pd.concat(res_tables)
        over_all_seeds.index.set_names('seed', level=0, inplace=True)
        over_all_seeds.to_csv(os.path.join(odir, "over_all_seeds.csv"))

