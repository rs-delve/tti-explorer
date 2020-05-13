import json

import numpy as np
import pandas as pd

from contacts import Contacts, NCOLS
from generate_cases import Case
from strategies import RETURN_KEYS


def results_table(results_dct, index_name="scenario"):
    df = {k:v.T for k, v in results_dct.items()}
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


# def run_scenario(case_contacts, strategy, rng, strategy_cgf_dct):
    # return pd.DataFrame([strategy(*cc, rng, **strategy_cgf_dct) for cc in case_contacts])


def run_scenario(case_contacts, strategy, rng, strategy_cgf_dct):
    df = pd.DataFrame([strategy(*cc, rng, **strategy_cgf_dct) for cc in case_contacts])

    df[RETURN_KEYS.percent_primary_symptomatic_missed] = df[RETURN_KEYS.num_primary_symptomatic_missed].sum() / df[RETURN_KEYS.num_primary_symptomatic].sum()
    df[RETURN_KEYS.percent_primary_asymptomatic_missed] = df[RETURN_KEYS.num_primary_asymptomatic_missed].sum() / df[RETURN_KEYS.num_primary_asymptomatic].sum()
    df[RETURN_KEYS.percent_primary_missed] = df[RETURN_KEYS.num_primary_missed].sum() / df[RETURN_KEYS.num_primary].sum()

    df[RETURN_KEYS.percent_secondary_from_symptomatic_missed] = df[RETURN_KEYS.num_secondary_from_symptomatic_missed].sum() / df[RETURN_KEYS.num_secondary_from_symptomatic].sum()
    df[RETURN_KEYS.percent_secondary_from_asymptomatic_missed] = df[RETURN_KEYS.num_secondary_from_asymptomatic_missed].sum() / df[RETURN_KEYS.num_secondary_from_asymptomatic].sum()
    df[RETURN_KEYS.percent_secondary_missed] = df[RETURN_KEYS.num_secondary_missed].sum() / df[RETURN_KEYS.num_secondary].sum()

    df.drop(columns=[
        RETURN_KEYS.num_primary_symptomatic,
        RETURN_KEYS.num_primary_asymptomatic,
        RETURN_KEYS.num_primary,
        RETURN_KEYS.num_primary_symptomatic_missed,
        RETURN_KEYS.num_primary_asymptomatic_missed,
        RETURN_KEYS.num_primary_missed,
        RETURN_KEYS.num_secondary_from_symptomatic,
        RETURN_KEYS.num_secondary_from_asymptomatic,
        RETURN_KEYS.num_secondary,
        RETURN_KEYS.num_secondary_from_symptomatic_missed,
        RETURN_KEYS.num_secondary_from_asymptomatic_missed,
        RETURN_KEYS.num_secondary_missed,
        ], inplace=True
    )

    return pd.concat({'mean': df.mean(0), 'std': df.std(0)}, axis=1)


def find_case_files(folder, ending=".json"):
    return list(filter(lambda x: x.endswith(ending), os.listdir(folder)))


def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)


def scale_results(results, monte_carlo_factor, r_monte_carlo_factor, nppl):
    rvals = [RETURN_KEYS.base_r, RETURN_KEYS.reduced_r]
    percentages = [
        RETURN_KEYS.percent_primary_symptomatic_missed,
        RETURN_KEYS.percent_primary_asymptomatic_missed,
        RETURN_KEYS.percent_primary_missed,
        RETURN_KEYS.percent_secondary_from_symptomatic_missed,
        RETURN_KEYS.percent_secondary_from_asymptomatic_missed,
        RETURN_KEYS.percent_secondary_missed,
    ]
    
    scale = []
    for k in results.index:
        if k in rvals:
            scale.append(1)
        elif k in percentages:
            scale.append(100)
        else:
            scale.append(nppl)
    
    scale = pd.Series(scale, index=results.index)
    
    results['mean'] = results['mean'] * scale
    results['std'] = results['std'] * scale

    mc_scale = []
    for k in results.index:
        if k in rvals:
            mc_scale.append(r_monte_carlo_factor)
        elif k in percentages:
            mc_scale.append(1)
        else:
            mc_scale.append(monte_carlo_factor)

    mc_std_error_factors = pd.Series(mc_scale, index=results.index)

    results['std'] = results['std'] * mc_std_error_factors

    return results


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
    import utils
    
    parser = ArgumentParser(fromfile_prefix_chars="@")
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
        help="Folder in which to save the outputs. Will be made for you if not specified.",
        type=str
        )
    parser.add_argument(
            "--scenarios",
            help=("Which scenarios to run from config.py. If 'all' then all are run. "
            "Default %(default)s."),
            default=[config.ALL_CFG_FLAG],
            type=str,
            nargs="*"
        )
    parser.add_argument(
            "--seed",
            help="Seed for random number generator. All runs will be re-seeded with this value. Default %(default)s.",
            default=0,
            type=int
        )
    args = parser.parse_args()

    strategy = registry[args.strategy]
    strategy_configs = config.get_strategy_config(
            args.strategy,
            args.scenarios
        )

    scenario_results = defaultdict(dict)
    case_files = find_case_files(args.population)
    pbar = tqdm(
            desc="Running strategies",
            total=len(case_files) * len(strategy_configs),
            smoothing=0
        )
    
    case_metadata = dict()
    for i, case_file in enumerate(case_files):
        case_contacts, metadata = load_cases(os.path.join(args.population, case_file))
        case_metadata[tidy_fname(case_file)] = metadata
        nppl = metadata['case_config']['infection_proportions']['nppl']
        rng = np.random.RandomState(seed=args.seed)

        n_monte_carlo_samples = len(case_contacts)
        n_r_monte_carlo_samples = len(case_contacts) * (metadata['case_config']['infection_proportions']['dist'][1] + metadata['case_config']['infection_proportions']['dist'][2])
        
        monte_carlo_factor = 1. / np.sqrt(n_monte_carlo_samples)
        r_monte_carlo_factor = 1. / np.sqrt(n_r_monte_carlo_samples)

        for scenario, cfg_dct in strategy_configs.items():
            r = run_scenario(
                    case_contacts,
                    strategy,
                    rng,
                    cfg_dct
                )
            scenario_results[scenario][tidy_fname(case_file)] = scale_results(
                    r,
                    monte_carlo_factor,
                    r_monte_carlo_factor,
                    nppl 
                )
                
            pbar.update(1)

    # dct = {k: pd.concat(v, axis=1).T for k, v in scenario_results.items()}
    # df = pd.concat(dct, axis=0) 

    tables = dict()
    os.makedirs(args.output_folder, exist_ok=True)
    for scenario, v in scenario_results.items():
        table = results_table(v, 'case_file')
        table.to_csv(
                os.path.join(
                    args.output_folder,
                    f"{scenario}.csv"
                )
            )
        tables[scenario] = table
    utils.write_json(case_metadata, os.path.join(args.output_folder, "case_metadata.json"))
    # df = pd.DataFrame.from_dict(tables, orient="index")
    # df.groupby(level=0).agg(['mean', 'std']).to_csv(os.path.join(args.output_folder, 'all_results.csv'))


    all_results = pd.concat(tables)
    all_results.index.set_names('scenario', level=0, inplace=True)

    all_results.to_csv(os.path.join(args.output_folder, 'all_results.csv'))
    all_results.unstack(level=-1).to_csv(os.path.join(args.output_folder, 'all_results_pivot.csv'))
