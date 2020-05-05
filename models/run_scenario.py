import json

import numpy as np

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
        cases_path="../data/cases/kucharski-cases.json",
        strategy="cmmid",
        scenarios="no_measures pop_testing".split(),
        seed=1,
        maxruns=50000,
        output_fpath=""
    )

    strategy = registry[args.strategy]
    strategy_configs = config.get_strategy_config(
            args.strategy,
            args.scenarios
        )

    case_contacts, metadata = load_cases(args.cases_path)

    rng = np.random.RandomState(seed=args.seed)
   
    results = dict()
    for scenario, cfg_dct in strategy_configs.items():
        scenario_outputs = list()

        start = time.time()
        for i, (case, contacts) in enumerate(case_contacts):
            if i == args.maxruns:
                break
            scenario_outputs.append(strategy(case, contacts, rng, **cfg_dct))

        scenario_outputs = np.array(scenario_outputs)
        results[scenario] = scenario_outputs
        print(scenario, scenario_outputs.mean(axis=0), f'took {time.time() - start:.1f}s')

    # can save this for later analysis
    outputs = dict(
            timestamp=datetime.now().strftime("%c"),
            results=results,
            case_metadata=metadata,
            args=args.__dict__
        )
