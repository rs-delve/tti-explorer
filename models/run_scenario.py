import json

import numpy as np
import pandas as pd

from contacts import Contacts, NCOLS
from generate_cases import Case


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


if __name__ == "__main__":
    from datetime import datetime
    import time
    from types import SimpleNamespace

    import config
    from strategies import registry

    args = SimpleNamespace(
        cases_paths=[
            "data/cases/kucharski_1.json",
            "data/cases/kucharski_2.json",
            "data/cases/kucharski_3.json",
            "data/cases/kucharski_4.json",
            "data/cases/kucharski_5.json",
            # "data/cases/oxteam_1.json",
            # "data/cases/oxteam_2.json",
            # "data/cases/oxteam_3.json",
            # "data/cases/oxteam_4.json",
            # "data/cases/oxteam_5.json",
        ],
        # strategy="cmmid",
        # scenarios=[
        #     'no_measures',
        #     'isolation_only',
        #     'hh_quaratine_only',
        #     'hh_work_only',
        #     'isolation_manual_tracing_met_limit',
        #     'isolation_manual_tracing_met_only',
        #     'isolation_manual_tracing',
        #     'cell_phone',
        #     'cell_phone_met_limit',
        #     'pop_testing',
        # ],
        # strategy="cmmid_better",
        # scenarios='all' # [
            # 'no_measures',
            # 'isolation_only',
            # 'hh_quarantine_only',
            # 'manual_tracing_work_only',
            # 'manual_tracing',
            # 'manual_tracing_limit_othr_contact',
            # 'manual_tracing_met_all_before',
            # 'manual_tracing_met_all_before_limit_othr_contact',
            # 'app_tracing',
            # 'app_tracing_met_limit',
            # 'both_tracing',
            # 'both_tracing_met_limit',
            # 'pop_testing',
            # 'all',
            # 'all_met_limit'
        # ],
        strategy="temporal_anne_flowchart",
        scenarios=[
            "no_measures",
            "default_scenario",
        ],
        seed=1,
        maxruns=50000,
        output_fpath=""
    )

    strategy = registry[args.strategy]
    strategy_configs = config.get_strategy_config(
            args.strategy,
            args.scenarios
        )


    rng = np.random.RandomState(seed=args.seed)
  
    results = dict()
    for j, cases_path in enumerate(args.cases_paths):

        case_contacts, metadata = load_cases(cases_path)

        for scenario, cfg_dct in strategy_configs.items():
            scenario_outputs = list()

            start = time.time()
            for i, (case, contacts) in enumerate(case_contacts):
                if i == args.maxruns:
                    break
                scenario_outputs.append(strategy(case, contacts, rng, **cfg_dct))

            scenario_outputs = np.array(scenario_outputs)
            results[scenario + f'-{j}'] = np.nanmean(scenario_outputs, axis=0)
            print(scenario, np.nanmean(scenario_outputs, axis=0), f'took {time.time() - start:.1f}s')


    results = pd.DataFrame.from_dict(results, orient='index', columns=['Base R', 'Reduced R', 'Manual Traces', 'App Traces', 'Tests Needed', 'PersonDays Quarantined', 'Wasted PersonDay Quarantined'])
    results.reset_index(inplace=True)

    results[['scenario', 'case set']] = results['index'].str.split("-", expand=True)
    results.drop(columns='index', inplace=True)

    results = results.groupby('scenario').agg(['mean', 'std'])

    results.to_csv('results.csv')

    print(results)

    # outputs = dict(
            # timestamp=datetime.now().strftime("%c"),
            # results=results,
            # case_metadata=metadata,
            # args=args.__dict__
        # )
    
    # # BE: method of  generating results table from dict in pandas
    # summary_df = pd.DataFrame.from_dict(
            # {k: v.mean(0) for k, v in results.items()},
            # orient='index',
            # columns=['Base R', 'Reduced R', 'Manual Tests']
        # )
    # print(summary_df)
