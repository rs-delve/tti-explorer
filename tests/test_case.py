from itertools import combinations_with_replacement

import numpy as np
import pytest

from tti_explorer.case import simulate_case, CaseFactors, Case

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


@pytest.mark.parametrize(
        "wfh,has_app,report_app,report_manual",
        combinations_with_replacement((True, False), 4)
        )
def test_case_factors(wfh, has_app, report_app, report_manual):
    return CaseFactors(
            wfh=wfh,
            has_app=has_app,
            report_app=report_app,
            report_manual=report_manual
        )


@pytest.mark.parametrize(
        ["case_under18", "case_symptomatic", "case_covid"],
        combinations_with_replacement((True, False), 3)
    )
@pytest.mark.parametrize(
        ['app_cov', 'go_to_school_prob', 'wfh_prob', 'compliance'],
        combinations_with_replacement((0, 1), 4)
        )
def test_simulate_case_factors(case_under18, case_symptomatic, case_covid,
        app_cov, go_to_school_prob, wfh_prob, compliance):
    rng = np.random.RandomState(seed=TEST_RANDOM_SEED)
    case = Case(
            under18=case_under18,
            symptomatic=case_symptomatic,
            covid=case_covid,
            day_noticed_symptoms=0,
            inf_profile=[0, 0, 1]
        )
    factors = CaseFactors.simulate_from(
        rng,
        case,
        app_cov,
        go_to_school_prob,
        wfh_prob,
        compliance
    )

    assert int(factors.wfh) == ((1 - go_to_school_prob) if case_under18 else wfh_prob)
    assert int(factors.has_app) == app_cov
    assert factors.report_app == (case_symptomatic and factors.has_app and bool(compliance))
    assert factors.report_manual == (case_symptomatic and (not factors.has_app) and bool(compliance))

