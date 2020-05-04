from collections import namedtuple

import numpy as np

from strategies import registry

Case = namedtuple(
        'Case',
        [
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


def simulate_case(rng, config):
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
                report_app=False
            )
    else:
        case_factors = dict(
                covid=illness == 2,
                symptomatic=True
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
    from config import CaseConfig
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('strategy', type=str)
    parser.add_argument('--nruns', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    
    case_config = CaseConfig()

    import time
    start = time.time()
    
    rng = np.random.RandomState(seed=args.seed)

    hparams = dict()
    strategy = registry[args.strategy]

    rs = np.zeros(args.nruns)
    for i in range(args.nruns):
        person = simulate_case(rng, case_config)
        rs[i] = strategy(person)

    print(f"Average r is {np.mean(rs)}")
    print(f"took {time.time() - start:.2f} seconds")


# p_leaves = [
        # p_asymptomatic_covid_pos,
        # p_symptomatic_covid_pos * p_has_app * p_report_app,
        # p_symptomatic_covid_pos * (1 - p_has_app) * 

        # ]

# leaves = [
    # Case(
        # covid=True,
        # symptomatic=False,
        # has_app=False,
        # report_app=False,
        # report_nhs=False
    # ),
    # Case(
        # covid=True,
        # symptomatic=True,
        # has_app=True,
        # report_app=True,
        # report_nhs=True
    # ),
    # Case(


    # ]



