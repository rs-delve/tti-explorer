from collections import namedtuple

import numpy as np


Case = namedtuple(
        'Case',
        [
            "over18",
            "covid",
            "symptomatic",
            "has_app",
            "report_nhs",
            "report_app",
        ]
    )


def bool_bernoulli(p, rng):
    return bool(rng.binomial(1, p))


def categorical(pvals, rng):
    return np.argwhere(rng.multinomial(1, pvals))


def simulate_case(config, rng):
    """simulate_case

    Simulate a single case.

    Args:
        rng:
        **hparams:

    Returns (Case): a case with various parameters

    Assumptions:
        - No double reporting
        - If asymptomatic and covid pos then don't have app
    """
    over18 = bool_bernoulli(config.p_over18, rng)

    illness_pvals = [
                config.p_asymptomatic_covid_pos,
                config.p_symptomatic_covid_neg,
                config.p_symptomatic_covid_pos,
            ]
    illness = categorical(illness_pvals, rng)

    if illness == 0:
        return Case(
                covid=True,
                symptomatic=False,
                has_app=False,
                report_nhs=False,
                report_app=False,
                over18=over18
            )
    else:
        case_factors = dict(
                covid=illness == 2,
                symptomatic=True,
                over18=over18
            )

        if bool_bernoulli(config.p_has_app, rng):
            case_factors["has_app"] = True
            if bool_bernoulli(config.p_report_app, rng):
                return Case(
                    report_app=True,
                    report_nhs=False,
                    **case_factors
                )
            else:
                return Case(
                        report_nhs=bool_bernoulli(
                            config.p_report_nhs_g_app,
                            rng
                        ),
                        report_app=False,
                        **case_factors
                    )
        else:
            case_factors["has_app"] = False
            return Case(
                    report_nhs=bool_bernoulli(
                        config.p_report_nhs_g_no_app,
                        rng
                    ),
                    report_app=False,
                    **case_factors
                )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from config import CaseConfig
    import os
    import time

    from contacts import EmpiricalContactsSimulator
    from strategies import registry

    parser = ArgumentParser()
    parser.add_argument('strategy', type=str)
    parser.add_argument('--nruns', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    infectivity_period = 5
    sars = dict(home_sar=0.2, work_sar=0.2, other_sar=0.2)

    start = time.time()

    case_config = CaseConfig()
    rng = np.random.RandomState(seed=args.seed)

    strategy = registry[args.strategy]

    data_folder = "../data"

    def load_csv(pth):
        return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")
    
    over18 = load_csv(os.path.join(data_folder, "contact_distributions_o18.csv"))
    under18 = load_csv(os.path.join(data_folder, "contact_distributions_u18.csv"))
 
    contacts_simulator = EmpiricalContactsSimulator(over18, under18, rng)

    rs = np.zeros(args.nruns)
    for i in range(args.nruns):
        case = simulate_case(case_config, rng)
        contacts = contacts_simulator(
                case,
                period=infectivity_period,
                **sars
            )
        rs[i] = strategy(case, contacts)

    print(f"Average r is {np.mean(rs)}")
    print(f"took {time.time() - start:.2f} seconds")


