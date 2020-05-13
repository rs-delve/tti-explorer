import json

import numpy as np
import pandas as pd

from .contacts import Contacts, NCOLS
from .generate_cases import Case


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
        case = Case(**dict(dict(inf_profile=None), **dct['case']))

        contacts_dct = dct['contacts']
        n_daily = contacts_dct.pop('n_daily')
        contacts_dct = {k: np.array(v, dtype=int).reshape(-1, NCOLS) for k, v in contacts_dct.items()}
        contacts = Contacts(n_daily=n_daily, **contacts_dct)
        pairs.append((case, contacts))
    return pairs, meta


def n_infected(contacts):
    return np.sum(contacts[:, 0] >= 0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("cases_path", type=str)
    args = parser.parse_args()
   
    # loads cases
    case_contacts, metadata = load_cases(args.cases_path)
    # case_contacts : list of (Case, Contacts) pairs
   
    n_covid = 0
    outputs = list()
    for case, contacts in case_contacts:
        if case.covid:
            n_covid += 1
            home = n_infected(contacts.home)
            work = n_infected(contacts.work)
            other = n_infected(contacts.other)
            total = home + work + other
            outputs.append((home, work, other, total))
    outputs = np.array(outputs)
    print("R values (mean). home: {:.2f} work: {:.2f} other: {:.2f} total: {:.2f}.".format(*outputs.mean(0)))
    print("R values  (std). home: {:.2f} work: {:.2f} other: {:.2f} total: {:.2f}.".format(*outputs.std(0)))
    n_cases = len(case_contacts)
    print(f"{n_covid}/{n_cases} = {n_covid / n_cases} have covid")

