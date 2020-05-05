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


def do_my_strategy(cases, contacts, rng, bar):
    pass


if __name__ == "__main__":
    cases_path = "../data/cases"
    rng = np.random.RandomState(seed=0)
   
    # loads cases
    case_contacts, metadata = load_cases(cases_path)
    # case_contacts : list of (Case, Contacts) pairs
   
    config = {
        "foo": "bar"
        }

   
    outputs = list()
    for case, contacts in case_contacts:
        output = do_my_strategy(case, contacts, rng, **config)
        outputs.append(output)
