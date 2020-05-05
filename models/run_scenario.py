from collections import namedtuple

import numpy as np


Case = namedtuple(
        'Case',
        [
            "under18",
            "covid",
            "symptomatic",
            "has_app",
            "report_nhs",
            "report_app",
            "day_noticed_symptoms",
        ]
    )


def bool_bernoulli(p, rng):
    return bool(rng.binomial(1, p))


def categorical(pvals, rng):
    return np.argwhere(rng.multinomial(1, pvals)).item()


def simulate_case(config, rng):
    """simulate_case
    Assumptions:
        - No double reporting
        - If asymptomatic and covid pos then don't have app
    """
    under18 = bool_bernoulli(config.p_under18, rng)

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
                under18=under18,
                day_noticed_symptoms=-1
            )
    else:
        case_factors = dict(
                covid=illness == 2,
                symptomatic=True,
                under18=under18,
                day_noticed_symptoms=categorical(config.p_day_noticed_symptoms, rng)
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
    import os
    import time
    from types import SimpleNamespace

    import config
    from contacts import EmpiricalContactsSimulator
    from strategies import registry

    start = time.time()
    case_config = config.get_case_config("kucharski")
    # Next line is a bit odd, will change case generation function to
    # take all parameters as args later.
    case_config = SimpleNamespace(**case_config)

    rng = np.random.RandomState(seed=1)
    runs=50000

    # strategy = registry[args.strategy]
    # # Change config to multiple 
    # strategy_config = config.get_strategy_config(
            # args.strategy,
            # args.strategy_config
            # )
    kucharski_scenarios = [
    "no_measures","isolation_only","hh_quaratine_only","hh_work_only",
                     "isolation_manual_tracing_met_only","isolation_manual_tracing_met_limit",
                     "isolation_manual_tracing","cell_phone","cell_phone_met_limit",
                     "pop_testing"]

    # kucharski_scenarios = ["isolation_manual_tracing_met_only"]

    data_folder = "data"

    def load_csv(pth):
        return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")
    
    over18 = load_csv(os.path.join(data_folder, "contact_distributions_o18.csv"))
    under18 = load_csv(os.path.join(data_folder, "contact_distributions_u18.csv"))
 
    contacts_simulator = EmpiricalContactsSimulator(over18, under18, rng)

    start = time.time()
    case_contacts = []
    for i in range(runs):
        case = simulate_case(case_config, rng)
        contacts = contacts_simulator(
                case,
                period=config.infectivity_period,
                home_sar=config.home_sar,
                work_sar=config.work_sar,
                other_sar=config.other_sar
            )
        case_contacts.append((case, contacts))
    print(f"Case generation took {time.time() - start:.2f} seconds")

    for scenario in kucharski_scenarios:
        outputs = list()
        start = time.time()

        strategy = registry["cmmid"]
        strategy_config = config.get_strategy_config(
                "cmmid",
                scenario,
                )

        for case, contacts in case_contacts:
            outputs.append(strategy(case, contacts, rng, **strategy_config))

        outputs = np.array(outputs)
        print(scenario, outputs.mean(axis=0), f'took {time.time() - start}s')
