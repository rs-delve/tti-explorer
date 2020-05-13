import numpy as np
import pytest

from tti_explorer import Case, Contacts
from tti_explorer.strategies import registry, RETURN_KEYS
from tti_explorer import config

TEST_RANDOM_SEED = 42


@pytest.mark.parametrize("has_covid", [True, False])
@pytest.mark.parametrize("s_level", list(config.S_levels.keys()))
@pytest.mark.parametrize("contact_trace_option", list(config.contact_trace_options.keys()))
def test_temporal_anne_flowchart_single_case_no_contacts(has_covid, s_level, contact_trace_option):
    case = Case(
        under18=False,
        covid=has_covid,
        symptomatic=False,
        day_noticed_symptoms=0,
        inf_profile=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    contacts = Contacts(0, np.empty((0,2), dtype=int), np.empty((0,2), dtype=int), np.empty((0,2), dtype=int))
    scenario = s_level + "_" + contact_trace_option

    try:
        parameters = config.get_strategy_config("temporal_anne_flowchart", [scenario])[scenario]
    except ValueError:
        assert s_level == "S0"
        return

    rng = np.random.RandomState(seed=TEST_RANDOM_SEED)

    strategy = registry['temporal_anne_flowchart']
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
    assert result[RETURN_KEYS.wasted_quarantine] == 0

