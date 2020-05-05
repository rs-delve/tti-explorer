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

# Add in notice symptoms
# change period to 5 days

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

    import config
    from config import CaseConfig
    from contacts import EmpiricalContactsSimulator
    from strategies import registry

    start = time.time()
    case_config = CaseConfig()
    rng = np.random.RandomState(seed=1)
    runs=50000

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
        print(scenario, np.nanmean(outputs, axis=0), f'took {time.time() - start}s')


# if __name__ == "__main__":
#     from argparse import ArgumentParser
#     import os
#     import time

#     import config
#     from config import CaseConfig
#     from contacts import EmpiricalContactsSimulator
#     from strategies import registry

#     parser = ArgumentParser()
#     parser.add_argument('strategy', type=str)
#     parser.add_argument('strategy_config', type=str)
#     parser.add_argument('--nruns', default=20, type=int)
#     parser.add_argument('--seed', default=0, type=int)
#     args = parser.parse_args()

#     start = time.time()
#     case_config = CaseConfig()
#     rng = np.random.RandomState(seed=args.seed)

#     strategy = registry[args.strategy]
#     strategy_config = config.get_strategy_config(
#             args.strategy,
#             args.strategy_config
#             )

#     data_folder = "data"

#     def load_csv(pth):
#         return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")
    
#     over18 = load_csv(os.path.join(data_folder, "contact_distributions_o18.csv"))
#     under18 = load_csv(os.path.join(data_folder, "contact_distributions_u18.csv"))
 
#     contacts_simulator = EmpiricalContactsSimulator(over18, under18, rng)

#     outputs = list()
#     for i in range(args.nruns):
#         case = simulate_case(case_config, rng)
#         contacts = contacts_simulator(
#                 case,
#                 period=config.infectivity_period,
#                 home_sar=config.home_sar,
#                 work_sar=config.work_sar,
#                 other_sar=config.other_sar
#             )
#         home_infections = (contacts.home[:, 0] >= 0).sum()  
#         work_infections = (contacts.work[:, 0] >= 0).sum() 
#         other_infections = (contacts.other[:, 0] >= 0).sum()

#         num_home = len(contacts.home[:, 0])
#         num_work = len(contacts.work[:, 0])
#         num_other = len(contacts.other[:, 0]) 

#         if num_home == 0: num_home += 1
#         if num_work == 0: num_work += 1
#         if num_other == 0: num_other += 1

#         outputs.append(np.array([home_infections/ num_home, work_infections / num_work, other_infections / num_other, home_infections + work_infections + other_infections]))
#         # outputs.append(strategy(case, contacts, rng, **strategy_config))
    
#     outputs = np.array(outputs)
#     print(np.nanmean(outputs, axis=0))
#     print(f"took {time.time() - start:.2f} seconds")
