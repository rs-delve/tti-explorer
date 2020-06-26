"""Generate case files"""

from argparse import ArgumentParser
from datetime import datetime
import json
import os


import numpy as np
from tqdm import trange


from tti_explorer import sensitivity
from tti_explorer.utils import ROOT_DIR
from tti_explorer.case_generator import get_generator_configs, CaseGenerator


def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")


def get_output_file_name(config_name, target, i, seed):
    if target is not None:
        return f"{config_name}_{target}{i}_seed{seed}.json"
    else:
        return f"{config_name}_seed{seed}.json"


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate JSON files of cases and contacts")
    parser.add_argument(
        "config_name",
        type=str,
        help="Name for config of cases and contacts. Will pull from config.py.",
    )
    parser.add_argument(
        "ncases", help="Number of cases w/ contacts to generate", type=int
    )
    parser.add_argument(
        "output_folder",
        help="Folder in which to store json files of cases and contacts",
        type=str,
    )
    parser.add_argument(
        "--seeds",
        help="random seeds for each population",
        default=-1,
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--sensitivity",
        help=(
            "Method for sensitivity analysis "
            "over parameters designated for sensitivity analysis in config.py. "
            "Empty string does no sensitivity analysis. Default '%(default)s'."
        ),
        default="",
        type=str,
    )
    parser.add_argument(
        "--n-pops",
        help="Number of i.i.d. populations to draw. Ignored if seeds is given.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(ROOT_DIR, "data", "bbc-pandemic"),
        type=str,
        help="Folder containing empirical tables of contact numbers. "
        "Two files are expected: contact_distributions_o18.csv and contact_distributions_u18.csv",
    )
    args = parser.parse_args()
    seeds = range(args.n_pops) if args.seeds == -1 else args.seeds

    os.makedirs(args.output_folder, exist_ok=True)

    case_configs, contacts_config = get_generator_configs(args.config_name, args.sensitivity)

    over18 = load_csv(os.path.join(args.data_dir, "contact_distributions_o18.csv"))
    under18 = load_csv(os.path.join(args.data_dir, "contact_distributions_u18.csv"))

    for i, dct in enumerate(case_configs):
        case_config = dct[sensitivity.CONFIG_KEY]
        target = dct[sensitivity.TARGET_KEY]
        print(target)

        for seed in seeds:
            case_generator = CaseGenerator(seed, over18, under18)

            cases_and_contacts = list()
            for _ in trange(args.ncases, smoothing=0, desc=f"Generating case set with seed {seed}."):
                output = case_generator.generate_case_with_contacts(case_config, contacts_config)
                cases_and_contacts.append(output)

            full_output = dict(
                timestamp=datetime.now().strftime("%c"),
                case_config=case_config,
                contacts_config=contacts_config,
                args=dict(args.__dict__, seed=seed),
                cases=cases_and_contacts,
            )

            fname = get_output_file_name(args.config_name, target, i, seed)
            with open(os.path.join(args.output_folder, fname), "w") as f:
                json.dump(full_output, f)
