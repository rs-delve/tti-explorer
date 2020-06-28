"""Print basic statistics of cases in a case file"""
from argparse import ArgumentParser

from tti_explorer import utils
from tti_explorer.case_statistics import CaseStatistics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "cases_path", type=str, help="File from which to load cases, should be .json."
    )
    args = parser.parse_args()

    case_contacts, metadata = utils.load_cases(args.cases_path)
    stats = CaseStatistics(case_contacts)

    print(
        "R values (mean). home: {:.2f} work: {:.2f} other: {:.2f} total: {:.2f}.".format(
            *(stats.mean_R)
        )
    )
    print(
        "R values  (std). home: {:.2f} work: {:.2f} other: {:.2f} total: {:.2f}.".format(
            *(stats.std_R)
        )
    )

    print(f"{stats.covid_count}/{stats.case_count} = {stats.covid_count / stats.case_count} have covid")
