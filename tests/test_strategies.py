import numpy as np
import pytest

from tti_explorer import Case, Contacts
from tti_explorer.contacts import EmpiricalContactsSimulator
from tti_explorer.strategies import registry, RETURN_KEYS
from tti_explorer import config


SIMULATION_LENGTH = 4


@pytest.mark.parametrize("has_covid", [True, False])
@pytest.mark.parametrize("s_level", list(config.S_levels.keys()))
@pytest.mark.parametrize("contact_trace_option", list(config.contact_trace_options.keys()))
def test_temporal_anne_flowchart_no_contacts(has_covid, s_level, contact_trace_option):
    case = Case(
        under18=False,
        covid=has_covid,
        symptomatic=False,
        day_noticed_symptoms=0,
        inf_profile=[0.0] * SIMULATION_LENGTH
    )
    contacts = Contacts(0, np.empty((0,2), dtype=int), np.empty((0,2), dtype=int), np.empty((0,2), dtype=int))
    scenario = s_level + "_" + contact_trace_option if s_level != "S0" else s_level

    parameters = config.get_strategy_configs("delve", [scenario])[scenario]

    rng = np.random.RandomState()

    strategy = registry['delve']
    result = strategy(case, contacts, rng, **parameters)

    if has_covid:
        result[RETURN_KEYS.base_r] == 0
        result[RETURN_KEYS.reduced_r] == 0
    else:
        assert np.isnan(result[RETURN_KEYS.base_r])
        assert np.isnan(result[RETURN_KEYS.reduced_r])
    assert result[RETURN_KEYS.man_trace] == 0
    assert result[RETURN_KEYS.app_trace] == 0
    assert result[RETURN_KEYS.tests] == 0
    assert result[RETURN_KEYS.quarantine] == 0



@pytest.mark.parametrize("s_level", list(config.S_levels.keys()))
@pytest.mark.parametrize("contact_trace_option", list(config.contact_trace_options.keys()))
def test_temporal_anne_flowchart_with_contacts(s_level, contact_trace_option):
    rng = np.random.RandomState()

    case = Case(
        under18=False,
        covid=True,
        symptomatic=True,
        day_noticed_symptoms=1,
        inf_profile=np.array([1.0] + [0.0] * (SIMULATION_LENGTH - 1))
    )

    over18_contacts_data = np.array([[1, 1, 1]])
    under18_contacts_data = np.array([[0, 0, 0]])
    contacts_simulator = EmpiricalContactsSimulator(over18_contacts_data, under18_contacts_data, rng)
    contacts = contacts_simulator(case, home_sar=0.1, work_sar=0.1, other_sar=0.1, asymp_factor=1, period=SIMULATION_LENGTH)

    scenario = s_level + "_" + contact_trace_option if s_level != "S0" else s_level
    parameters = config.get_strategy_configs("delve", [scenario])[scenario]

    strategy = registry['delve']
    result = strategy(case, contacts, rng, **parameters)

    assert 0 <= result[RETURN_KEYS.base_r] <= over18_contacts_data.sum()
    assert 0 <= result[RETURN_KEYS.reduced_r] <= over18_contacts_data.sum()
    if contact_trace_option == "no_TTI":
        assert result[RETURN_KEYS.tests] == 0
    else:
        assert 0 <= result[RETURN_KEYS.tests] <= over18_contacts_data.sum() + 1
