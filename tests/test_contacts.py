import numpy as np
import pytest

from tti_explorer import Case
from tti_explorer.contacts import EmpiricalContactsSimulator

TEST_RANDOM_SEED = 42
SIMULATION_LENGTH = 4


def create_case(under18):
    case = Case(
        under18=under18,
        covid=True,
        symptomatic=True,
        day_noticed_symptoms=0,
        inf_profile=np.ones((SIMULATION_LENGTH)) / SIMULATION_LENGTH
    )
    return case


@pytest.mark.parametrize('is_under18', [(True), (False)])
def test_contacts(is_under18):
    rng = np.random.RandomState(seed=TEST_RANDOM_SEED)
    over18 = np.array([[1, 1, 1]])
    under18 = np.array([[2, 2, 2]])
    contacts_simulator = EmpiricalContactsSimulator(over18, under18, rng)
    case = create_case(is_under18)

    contacts = contacts_simulator(case, home_sar=0.1, work_sar=0.1, other_sar=0.1, asymp_factor=1, period=SIMULATION_LENGTH)

    contact_data = under18 if is_under18 else over18
    assert contacts.home.shape[0] == contact_data[0][0]
    assert contacts.work.shape[0] == SIMULATION_LENGTH * contact_data[0][1]
    assert contacts.other.shape[0] == SIMULATION_LENGTH * contact_data[0][2]
