"""Generate case files"""

import numpy as np

from tti_explorer.case import simulate_case
from tti_explorer.utils import ROOT_DIR


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


def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")

if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    import json
    import os
    import time

    from tqdm import trange

    from tti_explorer import config, sensitivity
    from tti_explorer.contacts import EmpiricalContactsSimulator

    parser = ArgumentParser(description="Generate JSON files of cases and contacts")
    parser.add_argument(
            'config_name',
            type=str,
            help="Name for config of cases and contacts. Will pull from config.py."
            )
    parser.add_argument('ncases', help="Number of cases w/ contacts to generate", type=int)
    parser.add_argument('output_folder', help="Folder in which to store json files of cases and contacts", type=str)
    parser.add_argument(
            '--seeds',
            help="random seeds for each population",
            default=-1,
            type=int,
            nargs="*"
        )
    parser.add_argument(
            "--sensitivity",
            help=("Method for sensitivity analysis "
                "over parameters designated for sensitivity analysis in config.py. "
                "Empty string does no sensitivity analysis. Default '%(default)s'."),
            default="",
            type=str
        )
    parser.add_argument('--n-pops', help="Number of i.i.d. populations to draw. Ignored if seeds is given.", type=int, default=1)
    parser.add_argument(
            '--data-dir',
            default=os.path.join(ROOT_DIR, "data", "bbc-pandemic"),
            type=str,
            help="Folder containing empirical tables of contact numbers. "
                 "Two files are expected: contact_distributions_o18.csv and contact_distributions_u18.csv"
        )
    args = parser.parse_args()
    seeds = range(args.n_pops) if args.seeds == -1 else args.seeds
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    start = time.time()

    base_case_config = config.get_case_config(args.config_name)
    contacts_config = config.get_contacts_config(args.config_name)

    if args.sensitivity:
        config_generator = sensitivity.registry[args.sensitivity]
        cfgs = config_generator(base_case_config, config.get_case_sensitivities(args.config_name))
    else:
        cfgs = [{sensitivity.CONFIG_KEY: base_case_config, sensitivity.TARGET_KEY: None}]
    
    over18 = load_csv(os.path.join(args.data_dir, "contact_distributions_o18.csv"))
    under18 = load_csv(os.path.join(args.data_dir, "contact_distributions_u18.csv"))
    
    for i, dct in enumerate(cfgs):
        case_config = dct[sensitivity.CONFIG_KEY]
        for seed in seeds:
            rng = np.random.RandomState(seed=seed)
            contacts_simulator = EmpiricalContactsSimulator(over18, under18, rng)

            cases_and_contacts = list()
            for _ in trange(args.ncases, smoothing=0, desc=f"Generating case set with seed {seed}."):
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
                    args=dict(args.__dict__, seed=seed),
                    cases=cases_and_contacts
                )

            target = dct.get(sensitivity.TARGET_KEY)
            fname = (
                f"{args.config_name}_{target}{i}_seed{seed}.json"
                if target is not None else f"{args.config_name}_seed{seed}.json"
            )
            with open(os.path.join(args.output_folder, fname), "w") as f:
                json.dump(full_output, f)

