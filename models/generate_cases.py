from collections import namedtuple

import numpy as np

from utils import bool_bernoulli, categorical

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
            "inf_profile"
        ]
    )


def simulate_case(rng, p_under18, p_symptomatic_covid_neg,
        p_symptomatic_covid_pos, p_asymptomatic_covid_pos, p_has_app,
        p_report_app, p_report_nhs_g_app, p_report_nhs_g_no_app,
        p_day_noticed_symptoms, inf_profile):
    """simulate_case

    Args:
        rng (np.random.RandomState): random number generator.
        p_under18 (float): Probability of case being under 18
        p_symptomatic_covid_neg (float): Probability of being symptomatic and covid negative
        p_symptomatic_covid_pos (float): Probability of being symptomatic and covid positive
        p_asymptomatic_covid_pos (float): Probability of being asymptomatic and covid positive
        p_has_app (float): Probability of having app given symptomatic
        p_report_app (float): Probability of reporting through app conditional on having app
        p_report_nhs_g_app (float): Probability reporting with app given have app
        p_report_nhs_g_no_app (float): Probability of reporting through nhs given not have app
        p_day_noticed_symptoms (np.array[float]): Distribution of day on which case notices
            their symptoms. (In our model this is same as reporting symptoms.)
            Conditional on being symptomatic.
        inf_profile (list[float]): Probability of encounter with case
            on a given day causing a contact to be infected.

    Returns (Case): case with attributes populated.
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
                day_noticed_symptoms=-1,
                inf_profile=np.array(inf_profile)
            )
    else:
        covid = illness == 2
        profile = np.array(inf_profile) if covid else np.zeros(len(inf_profile))
        case_factors = dict(
                covid=covid,
                symptomatic=True,
                under18=under18,
                day_noticed_symptoms=categorical(p_day_noticed_symptoms, rng),
                inf_profile=profile
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


def case_as_dict(case):
    dct = case._asdict()
    dct['inf_profile'] = dct['inf_profile'].tolist()
    return dct


def contacts_as_dict(contacts):
    contacts_dct = {
            k:
            v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in contacts._asdict().items()
        }
    contacts_dct['n_daily'] = {k: int(v) for k,v in contacts_dct['n_daily'].items()}
    return contacts_dct


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
        output['case'] = case_as_dict(case)
        output['contacts'] = contacts_as_dict(contacts)
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
