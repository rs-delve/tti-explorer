import numpy as np

from . import config, sensitivity
from .case import simulate_case
from .contacts import EmpiricalContactsSimulator
from .utils import CASE_KEY, CONTACTS_KEY


def _case_as_dict(case):
    dct = case.to_dict()
    dct["inf_profile"] = dct["inf_profile"].tolist()
    return dct


def _contacts_as_dict(contacts):
    contacts_dct = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in contacts.to_dict().items()
    }
    contacts_dct["n_daily"] = {k: int(v) for k, v in contacts_dct["n_daily"].items()}
    return contacts_dct


def get_generator_configs(config_name, sensitivity_method):
    if not config_name:
        raise ValueError("config_name cannot be empty")

    base_case_config = config.get_case_config(config_name)
    contacts_config = config.get_contacts_config(config_name)

    if sensitivity_method:
        config_generator = sensitivity.registry[sensitivity_method]
        case_configs = config_generator(
            base_case_config, config.get_case_sensitivities(config_name)
        )
    else:
        case_configs = [
            {sensitivity.CONFIG_KEY: base_case_config, sensitivity.TARGET_KEY: None}
        ]

    return case_configs, contacts_config


class CaseGenerator():
    def __init__(self, random_seed, over18_contacts, under18_contacts):
        self.rng = np.random.RandomState(seed=random_seed)
        self.contacts_simulator = EmpiricalContactsSimulator(over18_contacts, under18_contacts, self.rng)

    def generate_case_with_contacts(self, case_config, contacts_config):
        case = simulate_case(self.rng, **case_config)
        contacts = self.contacts_simulator(case, **contacts_config)
        output = {
            CASE_KEY: _case_as_dict(case),
            CONTACTS_KEY: _contacts_as_dict(contacts)
        }

        return output
