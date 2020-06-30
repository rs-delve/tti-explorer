from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


from tti_explorer import config, sensitivity, strategies, utils
from tti_explorer.scenario import get_monte_carlo_factors, run_scenario, results_table, scale_results


if __name__ == "__main__":
    parser = ArgumentParser("Run sensitivity analysis on strategy")
    parser.add_argument("strategy", help="The name of the strategy to use", type=str)
    parser.add_argument(
        "population",
        help=(
            "Folder containing population files, "
            "we will assume all .json files in folder are to be used."
        ),
        type=str,
    )
    parser.add_argument(
        "output_folder",
        help="Folder in which to save the outputs. Will be made for you if needed.",
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
        "--sensitivity",
        help=(
            "Method for sensitivity analysis "
            "over parameters designated for sensitivity analysis in config.py. "
            "Empty string does no sensitivity analysis. Default '%(default)s'."
        ),
        default="",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="Seed for random number generator. All runs will be re-seeded with this value. Default %(default)s.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--parameters",
        help="Specific parameters to ablate over. Optional, if not present runs over all parameters defined for a strategy.",
        nargs="*",
    )
    parser.add_argument(
        "--nprocs",
        help="Number of cores on which to run this script. Default %(default)s",
        default=1,
        type=int,
    )
    args = parser.parse_args()
    strategy = strategies.registry[args.strategy]
    strategy_configs = config.get_strategy_configs(args.strategy, args.scenarios)
    config_generator = sensitivity.SensitivityConfigGenerator(args.sensitivity, args.parameters)

    case_files = utils.find_case_files(args.population)
    scenario_results = defaultdict(lambda: defaultdict(dict))
    configs_dct = defaultdict(lambda: defaultdict(dict))
    with ProcessPoolExecutor(max_workers=args.nprocs) as executor:
        futures = list()
        for case_file in tqdm(case_files, desc="loading cases"):
            case_contacts, metadata = utils.load_cases(
                os.path.join(args.population, case_file)
            )

            _, symp_covid_pos, asymp_covid_pos = metadata["case_config"]["infection_proportions"]["dist"]
            monte_carlo_factor, r_monte_carlo_factor = get_monte_carlo_factors(len(case_contacts), symp_covid_pos, asymp_covid_pos)

            nppl = metadata["case_config"]["infection_proportions"]["nppl"]
            for scenario, cfg_dct in strategy_configs.items():
                cfgs = config_generator.generate_for_strategy(args.strategy, cfg_dct)

                for i, cfg in enumerate(cfgs):
                    future = executor.submit(
                        run_scenario,
                        case_contacts,
                        strategy,
                        np.random.RandomState(seed=args.seed),
                        cfg[sensitivity.CONFIG_KEY],
                    )
                    futures.append(
                        (
                            (scenario, case_file, cfg),
                            (future, monte_carlo_factor, r_monte_carlo_factor, nppl),
                        )
                    )
                    # configs_dct[scenario][i][case_file] = cfg

        pbar = tqdm(
            desc="Running configurations/sensitivities",
            total=len(case_files)
            * len(strategy_configs)
            * 3
            * 3
            * 10,  # this is just number of entries in temporal anne sensitivities generator
            smoothing=None,
        )
        for i, ((scenario, case_file, cfg), arguments) in enumerate(futures):
            # this is so uglY!
            future = arguments[0]
            scenario_results[scenario][i][utils.tidy_fname(case_file)] = (
                scale_results(future.result(), *arguments[1:]),
                cfg,
            )
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
                    strategy=args.strategy,
                ),
                os.path.join(odir, f"config_{i}.json"),
            )

            table = results_table(res_dct_over_cases)
            table.to_csv(os.path.join(odir, f"run_{i}.csv"))
            res_tables[i] = table
        over_all_seeds = pd.concat(res_tables)
        over_all_seeds.index.set_names("seed", level=0, inplace=True)
        over_all_seeds.to_csv(os.path.join(odir, "over_all_seeds.csv"))
