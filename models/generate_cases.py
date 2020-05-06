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
            # "infectivity_profile"
        ]
    )


def bool_bernoulli(p, rng):
    return bool(rng.binomial(1, p))


def categorical(pvals, rng):
    return np.argwhere(rng.multinomial(1, pvals)).item()

"""
TODO BE:
    - change daily infectivity to draw from dist
    - increase length of simulation
    - use infectivity profile to generate contacts
    - add new config entry
"""

def simulate_case(
        rng,
        p_under18,
        p_symptomatic_covid_neg,
        p_symptomatic_covid_pos,
        p_asymptomatic_covid_pos,
        # Conditional on symptomatic
        p_has_app,
        # Conditional on having app
        p_report_app,
        p_report_nhs_g_app,
        # Conditional on not having app
        p_report_nhs_g_no_app,
        # Distribution of day on which the case notices their symptoms
        # This is conditinal on them being symptomatic at all
        p_day_noticed_symptoms
    ):
    """simulate_case
    Assumptions:
        - No double reporting
        - If asymptomatic and covid pos then don't have app
    """
    under18 = bool_bernoulli(p_under18, rng)

    illness_pvals = [
                p_asymptomatic_covid_pos,
                p_symptomatic_covid_neg,
                p_symptomatic_covid_pos,
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
                day_noticed_symptoms=categorical(p_day_noticed_symptoms, rng)
            )

        if bool_bernoulli(p_has_app, rng):
            case_factors["has_app"] = True
            if bool_bernoulli(p_report_app, rng):
                return Case(
                    report_app=True,
                    report_nhs=False,
                    **case_factors
                )
            else:
                return Case(
                        report_nhs=bool_bernoulli(
                            p_report_nhs_g_app,
                            rng
                        ),
                        report_app=False,
                        **case_factors
                    )
        else:
            case_factors["has_app"] = False
            return Case(
                    report_nhs=bool_bernoulli(
                        p_report_nhs_g_no_app,
                        rng
                    ),
                    report_app=False,
                    **case_factors
                )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    import json
    import os
    import time

    import config
    from contacts import EmpiricalContactsSimulator

    parser = ArgumentParser()
    parser.add_argument(
            'config_name',
            type=str,
            help="Name for config of cases and contacts. Will pull from config.py."
            )
    parser.add_argument('ncases', help="Number of cases w/ contacts to generate", type=int)
    parser.add_argument('output', help="json file in which to store cases and contacts", type=str)
    parser.add_argument('--seed', help="random seed", default=0, type=int)
    parser.add_argument(
            '--data-dir',
            default="../data",
            type=str,
            help="Folder containing empirical tables of contact numbers"
        )
    args = parser.parse_args()

    start = time.time()

    case_config = config.get_case_config(args.config_name)
    contacts_config = config.get_contacts_config(args.config_name)
    rng = np.random.RandomState(seed=args.seed)
    
    def load_csv(pth):
        return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")
    
    over18 = load_csv(os.path.join(args.data_dir, "contact_distributions_o18.csv"))
    under18 = load_csv(os.path.join(args.data_dir, "contact_distributions_u18.csv"))
    
    contacts_simulator = EmpiricalContactsSimulator(over18, under18, rng)

    cases_and_contacts = list()
    for i in range(args.ncases):
        case = simulate_case(rng, **case_config)
        contacts = contacts_simulator(
                case,
                **contacts_config
            )
        output = dict()
        output['case'] = case._asdict()
        contacts_dct = {
                k:
                v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in contacts._asdict().items()
            }
        contacts_dct['n_daily'] = {k: int(v) for k,v in contacts_dct['n_daily'].items()}
        output['contacts'] = contacts_dct
        cases_and_contacts.append(output)

    full_output = dict(
            timestamp=datetime.now().strftime('%c'),
            case_config=case_config,
            contacts_config=contacts_config,
            args=args.__dict__,
            cases=cases_and_contacts
        )

    with open(args.output, "w") as f:
        json.dump(full_output, f)

    print(f"Case and contact generation took {time.time() - start:.2f} seconds\n")
