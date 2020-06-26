import numpy as np
import pytest

from tti_explorer.case_generator import get_generator_configs, CaseGenerator
from tti_explorer import sensitivity
from tti_explorer.utils import CASE_KEY, CONTACTS_KEY
from tti_explorer.case import Case
from tti_explorer.contacts import Contacts

TEST_RANDOM_SEED = 42
CASE_CONFIG_NAME = "delve"


def test_generator_configs_normal():
    cases_configs, contacts_config = get_generator_configs(CASE_CONFIG_NAME, "grid")

    assert len(list(cases_configs)) > 1
    assert contacts_config is not None


def test_generator_configs_no_sensitivity():
    cases_configs, contacts_config = get_generator_configs(CASE_CONFIG_NAME, "")
    assert len(list(cases_configs)) == 1

    cases_configs, contacts_config = get_generator_configs(CASE_CONFIG_NAME, None)
    assert len(list(cases_configs)) == 1


def test_generator_configs_no_config():
    with pytest.raises(ValueError):
        get_generator_configs("", "")
    
    with pytest.raises(ValueError):
        get_generator_configs(None, "")


def test_case_generator():
    over18 = np.array([[1, 1, 1]])
    under18 = np.array([[2, 2, 2]])
    case_configs, contacts_config = get_generator_configs(CASE_CONFIG_NAME, "grid")
    case_config = next(case_configs)[sensitivity.CONFIG_KEY]

    case_generator = CaseGenerator(TEST_RANDOM_SEED, over18, under18)
    case_with_contacts = case_generator.generate_case_with_contacts(case_config, contacts_config)

    assert isinstance(case_with_contacts[CASE_KEY], dict)
    assert isinstance(case_with_contacts[CONTACTS_KEY], dict)

