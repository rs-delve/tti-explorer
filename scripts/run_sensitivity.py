import json

import numpy as np
import pandas as pd

from run_scenarios import run_scenario


def results_table(results_dct, index_name="scenario"):
    df = pd.concat({k: v.T for k, v in results_dct.items()})
    df.index.names = [index_name, config.STATISTIC_COLNAME]
    return df


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
    
    from tti_explorer import config, sensitivity, strategies, utils

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
                "we will assume all .json files in folder are to be used."),
            type=str
        )
    parser.add_argument(
            "output_folder",
            help="Folder in which to save the outputs. Will be made for you if needed.",
            type=str
        )
    parser.add_argument(
            "--scenarios",
            help="Which scenarios to run from config.py."
            "By default, if no value is given, all scenarios available for a given strategy are run.",
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
    strategy_configs = config.get_strategy_configs(
            args.strategy,
            args.scenarios
        )
    config_generator = sensitivity.registry[args.sensitivity] if args.sensitivity else None

    case_files = find_case_files(args.population)
    scenario_results = defaultdict(lambda: defaultdict(dict))
    configs_dct = defaultdict(lambda: defaultdict(dict))
    with ProcessPoolExecutor(max_workers=args.nprocs) as executor:
        futures = list()
        for case_file in tqdm(case_files, desc="loading cases"):
            case_contacts, metadata = utils.load_cases(os.path.join(args.population, case_file))

            # Can we turn this into something like calculate_confidence_interval?
            nppl = metadata['case_config']['infection_proportions']['nppl']
            n_monte_carlo_samples = len(case_contacts)
            n_r_monte_carlo_samples = len(case_contacts) * (
                    metadata['case_config']['infection_proportions']['dist'][1]
                    + metadata['case_config']['infection_proportions']['dist'][2]
                )

            monte_carlo_factor = 1. / np.sqrt(n_monte_carlo_samples)
            r_monte_carlo_factor = 1. / np.sqrt(n_r_monte_carlo_samples)

            n_monte_carlo_samples = len(case_contacts)
            n_r_monte_carlo_samples = len(case_contacts) * (
                    metadata['case_config']['infection_proportions']['dist'][1]
                    + metadata['case_config']['infection_proportions']['dist'][2]
                    )
            
            monte_carlo_factor = 1. / np.sqrt(n_monte_carlo_samples)
            r_monte_carlo_factor = 1. / np.sqrt(n_r_monte_carlo_samples)
            #

            for scenario, cfg_dct in strategy_configs.items():
                policy_sensitivities = config.get_policy_sensitivities(args.strategy)
                if args.parameters is not None:
                    policy_sensitivities = dict((k, policy_sensitivities[k]) for k in args.parameters)

                cfgs = config_generator(
                            cfg_dct,
                            policy_sensitivities
                        ) if args.sensitivity else [{sensitivity.CONFIG_KEY: cfg_dct, sensitivity.TARGET_KEY: ""}]

                for i, cfg in enumerate(cfgs):
                    future = executor.submit(
                            run_scenario,
                            case_contacts,
                            strategy,
                            np.random.RandomState(seed=args.seed),
                            cfg[sensitivity.CONFIG_KEY]
                        )
                    futures.append(
                            ((scenario, case_file, cfg), (future, monte_carlo_factor, r_monte_carlo_factor, nppl))
                            )
                    # configs_dct[scenario][i][case_file] = cfg

        pbar = tqdm(
            desc="Running configurations/sensitivities",
            total=len(case_files) * len(strategy_configs) * 3 * 3 * 10,  # this is just number of entries in temporal anne sensitivities generator
            smoothing=None
        )
        for i, ((scenario, case_file, cfg), arguments) in enumerate(futures):
            # this is so uglY!
            future = arguments[0]
            scenario_results[scenario][i][tidy_fname(case_file)] = scale_results(
                future.result(),
                *arguments[1:]
            ), cfg
            pbar.update(1)

    os.makedirs(args.output_folder, exist_ok=True)
    for scenario, res_dict in scenario_results.items():
        odir = os.path.join(args.output_folder, scenario)
        os.makedirs(odir, exist_ok=True)
        res_tables = dict()
        for i, dct in res_dict.items():
            cfg = next(iter(dct.values()))[1]
            res_dct_over_cases = {k: v[0] for k, v in dct.items()}
            utils.write_json(
                    dict(
                        cfg,
                        seed=args.seed,
                        population=args.population,
                        strategy=args.strategy
                    ),
                    os.path.join(odir, f"config_{i}.json")
                )

            table = results_table(res_dct_over_cases)
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
