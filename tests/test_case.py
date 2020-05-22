import numpy as np
import pytest

from tti_explorer.case import simulate_case

TEST_RANDOM_SEED = 42


@pytest.fixture
def case_config():
    return {
        'p_under18': False,
        'infection_proportions': {'dist': [0, 0, 1]},
        'p_day_noticed_symptoms': [0.5, 0.5],
        'inf_profile': [0.5, 0.5]
    }


@pytest.mark.parametrize("under18_prob,under18", [(0, False), (1, True)])
def test_simulate_case_under18(case_config, under18_prob, under18):
    rng = np.random.RandomState(seed=TEST_RANDOM_SEED)
    case_config['p_under18'] = under18_prob

    test_case = simulate_case(rng, **case_config)

    assert test_case.under18 == under18


@pytest.mark.parametrize("is_covid,symptomatic", [(True, True), (True, False), (False, True)])
def test_simulate_covid_symptomatic(case_config, is_covid, symptomatic):
    rng = np.random.RandomState(seed=TEST_RANDOM_SEED)
    if is_covid:
        if symptomatic:
            case_config['infection_proportions']['dist'] = [0, 1, 0]
        else:
            case_config['infection_proportions']['dist'] = [0, 0, 1]
    else:
        case_config['infection_proportions']['dist'] = [1, 0, 0]

    test_case = simulate_case(rng, **case_config)

    assert test_case.covid == is_covid
    assert test_case.symptomatic == symptomatic
