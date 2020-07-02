import numpy as np
import pandas as pd
import pytest

from tti_explorer import Case, Contacts
from tti_explorer.scenario import get_monte_carlo_factors, run_scenario, STATS_KEYS, scale_results, results_table
from tti_explorer.strategies import registry, RETURN_KEYS


def test_get_monte_carlo_factors():
    monte_carlo_factor, r_monte_carlo_factor = get_monte_carlo_factors(1, 0.125, 0.125)

    assert monte_carlo_factor == 1.0
    assert r_monte_carlo_factor == 2.0


def test_run_scenario():
    case = Case(
        covid=True,
        symptomatic=True,
        under18=False,
        day_noticed_symptoms=1,
        inf_profile=[0.5, 0.5],
    )
    contacts = Contacts(1, np.ones((1,2), dtype=int), np.ones((1,2), dtype=int), np.ones((1,2), dtype=int))

    def mock_strategy(*args, **kwargs):
        return {
            RETURN_KEYS.base_r: 1 if case.covid else np.nan,
            RETURN_KEYS.reduced_r: 2 if case.covid else np.nan,
            RETURN_KEYS.man_trace: 3,
            RETURN_KEYS.app_trace: 4,
            RETURN_KEYS.tests: 5,
            RETURN_KEYS.quarantine: 6,
            RETURN_KEYS.covid: case.covid,
            RETURN_KEYS.symptomatic: case.symptomatic,
            RETURN_KEYS.tested: True,
            RETURN_KEYS.secondary_infections: 7,
            RETURN_KEYS.cases_prevented_social_distancing: 8,
            RETURN_KEYS.cases_prevented_symptom_isolating: 9,
            RETURN_KEYS.cases_prevented_contact_tracing: 10,
            RETURN_KEYS.fractional_r: 11,
        }

    scenario_output = run_scenario([(case, contacts)], mock_strategy, np.random.RandomState, {})

    assert STATS_KEYS.mean in scenario_output.columns
    assert STATS_KEYS.std in scenario_output.columns
    assert scenario_output.loc[RETURN_KEYS.base_r][STATS_KEYS.mean] == 1
    assert scenario_output.loc[RETURN_KEYS.reduced_r][STATS_KEYS.mean] == 2


def test_scale_results():
    mock_results = pd.DataFrame({
        STATS_KEYS.mean: [1, 1, 1],
        STATS_KEYS.std: [1, 1, 1],
    }, index=[RETURN_KEYS.base_r, RETURN_KEYS.secondary_infections, RETURN_KEYS.percent_primary_missed])

    monte_carlo_factor=2
    r_monte_carlo_factor=3
    nppl=10
    scaled_results = scale_results(mock_results, monte_carlo_factor, r_monte_carlo_factor, nppl)

    assert scaled_results.loc[RETURN_KEYS.base_r][STATS_KEYS.mean] == \
        mock_results.loc[RETURN_KEYS.base_r][STATS_KEYS.mean]
    assert scaled_results.loc[RETURN_KEYS.base_r][STATS_KEYS.std] == \
        mock_results.loc[RETURN_KEYS.base_r][STATS_KEYS.std] * r_monte_carlo_factor
    assert scaled_results.loc[RETURN_KEYS.percent_primary_missed][STATS_KEYS.mean] == \
        mock_results.loc[RETURN_KEYS.percent_primary_missed][STATS_KEYS.mean] * 100
    assert scaled_results.loc[RETURN_KEYS.percent_primary_missed][STATS_KEYS.std] == \
        mock_results.loc[RETURN_KEYS.percent_primary_missed][STATS_KEYS.std] * 100
    assert scaled_results.loc[RETURN_KEYS.secondary_infections][STATS_KEYS.mean] == \
        mock_results.loc[RETURN_KEYS.secondary_infections][STATS_KEYS.mean] * nppl
    assert scaled_results.loc[RETURN_KEYS.secondary_infections][STATS_KEYS.std] == \
        mock_results.loc[RETURN_KEYS.secondary_infections][STATS_KEYS.std] * nppl * monte_carlo_factor
