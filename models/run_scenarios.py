import json

import numpy as np
import pandas as pd

from contacts import Contacts, NCOLS
from generate_cases import Case


def results_table(results_dct, index_name="scenario"):
    df = pd.DataFrame.from_dict(
            results_dct,
            orient="index"
        ).sort_index()
    df.index.name = index_name
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
    return pd.DataFrame([strategy(*cc, rng, **strategy_cgf_dct) for cc in case_contacts])


def find_case_files(folder, ending=".json"):
    return filter(lambda x: x.endswith(ending), os.listdir(folder))


def tidy_fname(fname, ending=".json"):
    return fname.rstrip(ending)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from collections import defaultdict
    from datetime import datetime
    import time
    from types import SimpleNamespace
    import os

    import config
    from strategies import registry
    
    parser = ArgumentParser()
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
            default=["all"],
            nargs='*',
            type=str
        )
    parser.add_argument(
            "--ablate",
            help="Whether to ablate over parameters designated for ablation in config.py. Default %(default)s.",
            action="store_true"
        )
    parser.add_argument(
            "--seed",
            help="Seed for random number generator. All runs will be re-seeded with this value. Default %(default)s.",
            default=0,
            type=int
        )
    args = parser.parse_args()

    # remove all this when dict output fo stratefies is done

    strategy = registry[args.strategy]
    strategy_configs = config.get_strategy_config(
            args.strategy,
            args.scenarios
        )

    scenario_results = defaultdict(dict)
    case_files = list(find_case_files(args.population))
    for i, case_file in enumerate(case_files):
        case_contacts, metadata = load_cases(os.path.join(args.population, case_file))
        rng = np.random.RandomState(seed=args.seed)

        for j, (scenario, cfg_dct) in enumerate(strategy_configs.items()):
            scenario_results[scenario][tidy_fname(case_file)] = run_scenario(
                    case_contacts,
                    strategy,
                    rng,
                    cfg_dct
                ).mean(0)
            print(f'Case set {i+1}/{len(case_files)}, strategy {j+1}/{len(strategy_configs)}')

    os.makedirs(args.output_folder, exist_ok=True)
    for k, v in scenario_results.items():
        results_table(v).to_csv(
                os.path.join(
                    args.output_folder,
                    f"{k}.csv"
                )
            )

    all_results = pd.DataFrame.from_dict({ (i,j): scenario_results[i][j] for i in scenario_results.keys() for j in scenario_results[i].keys()}, orient='index') 
    all_results.reset_index(inplace=True)
    all_results.columns = ['scenario', 'case_set'] + list(all_results.columns[2:])
    all_results = all_results.groupby('scenario').agg(['mean', 'std'])

    all_results.to_csv(os.path.join(args.output_folder, 'all.csv'))

        # TODO:
        # Save all configs and arguments
        # Implement ablation 
        # Implement population ablations (?)
