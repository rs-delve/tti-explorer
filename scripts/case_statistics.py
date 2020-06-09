"""Print basic statistics of cases in a case file"""
import numpy as np


def n_infected(contacts):
    return np.sum(contacts[:, 0] >= 0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from tti_explorer import utils

    parser = ArgumentParser()
    parser.add_argument(
        "cases_path", type=str, help="File from which to load cases, should be .json."
    )
    args = parser.parse_args()

    case_contacts, metadata = utils.load_cases(args.cases_path)
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
    print(
        "R values (mean). home: {:.2f} work: {:.2f} other: {:.2f} total: {:.2f}.".format(
            *outputs.mean(0)
        )
    )
    print(
        "R values  (std). home: {:.2f} work: {:.2f} other: {:.2f} total: {:.2f}.".format(
            *outputs.std(0)
        )
    )
    n_cases = len(case_contacts)
    print(f"{n_covid}/{n_cases} = {n_covid / n_cases} have covid")
