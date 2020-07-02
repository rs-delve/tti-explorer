"""
Contains methods that facilitate single scenario run
as well as handling results of that run.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import config
from .strategies import RETURN_KEYS


STATS_KEYS = SimpleNamespace(mean="mean", std="std")


def get_monte_carlo_factors(n_cases, symp_covid_pos, asymp_covid_pos):
    """
    Get Monte Carlo scale factors

    :param n_cases: Number of cases
    :param symp_covid_pos: Probability of case being symptomatic COVID positive
    :param asymp_covid_pos: Probability of case being asymptomatic COVID positive
    :returns: TODO
    """
    n_monte_carlo_samples = n_cases
    n_r_monte_carlo_samples = n_cases * (symp_covid_pos + asymp_covid_pos)

    monte_carlo_factor = 1.0 / np.sqrt(n_monte_carlo_samples)
    r_monte_carlo_factor = 1.0 / np.sqrt(n_r_monte_carlo_samples)

    return monte_carlo_factor, r_monte_carlo_factor


def run_scenario(case_contacts, strategy, rng, strategy_cgf_dct):
    """
    Run a single scenario

    :param case_contacts: List of tuples (Case, Contacts) to run scenario for
    :param strategy: TTI strategy to use
    :param rng: Random number generator
    :param strategy_cgf_dct: Dictionary of strategy parameter values
    :returns: Pandas dataframe with two rows, which represent mean and standard deviation of scenario outputs
    """
    df = pd.DataFrame([strategy(*cc, rng, **strategy_cgf_dct) for cc in case_contacts])

    positive_symptomatic_tested = (
        df[RETURN_KEYS.covid] & df[RETURN_KEYS.symptomatic] & df[RETURN_KEYS.tested]
    )
    positive_symptomatic_not_tested = (
        df[RETURN_KEYS.covid] & df[RETURN_KEYS.symptomatic] & ~df[RETURN_KEYS.tested]
    )
    positive_asymptomatic = df[RETURN_KEYS.covid] & ~df[RETURN_KEYS.symptomatic]
    # positive = df[RETURN_KEYS.covid]

    r_stopped_by_social_distancing = df[
        RETURN_KEYS.cases_prevented_social_distancing
    ].sum()

    r_stopped_by_social_distancing_positive_compliant = df[positive_symptomatic_tested][
        RETURN_KEYS.cases_prevented_social_distancing
    ].sum()

    r_stopped_by_symptom_isolating = df[positive_symptomatic_tested][
        RETURN_KEYS.cases_prevented_symptom_isolating
    ].sum()

    r_stopped_by_contact_tracing = df[positive_symptomatic_tested][
        RETURN_KEYS.cases_prevented_contact_tracing
    ].sum()

    r_remaining_asymptomatic = df[positive_asymptomatic][RETURN_KEYS.reduced_r].sum()
    r_remaining_positive_non_compliant = df[positive_symptomatic_not_tested][
        RETURN_KEYS.reduced_r
    ].sum()
    r_remaining_positive_compliant = df[positive_symptomatic_tested][
        RETURN_KEYS.reduced_r
    ].sum()

    num_secondary_cases = df[RETURN_KEYS.secondary_infections].sum()

    df[RETURN_KEYS.stopped_by_social_distancing_percentage] = (
        r_stopped_by_social_distancing / num_secondary_cases
    )
    df[RETURN_KEYS.stopped_by_symptom_isolating_percentage] = (
        r_stopped_by_symptom_isolating / num_secondary_cases
    )
    df[RETURN_KEYS.stopped_by_tracing_percentage] = (
        r_stopped_by_contact_tracing / num_secondary_cases
    )
    df[RETURN_KEYS.not_stopped_asymptomatic_percentage] = (
        r_remaining_asymptomatic / num_secondary_cases
    )
    df[RETURN_KEYS.not_stopped_symptomatic_non_compliant_percentage] = (
        r_remaining_positive_non_compliant / num_secondary_cases
    )
    df[RETURN_KEYS.not_stopped_by_tti_percentage] = (
        r_remaining_positive_compliant / num_secondary_cases
    )

    df[RETURN_KEYS.stopped_by_social_distancing_symptomatic_compliant_percentage] = (
        r_stopped_by_social_distancing_positive_compliant / num_secondary_cases
    )

    df.drop(
        columns=[
            RETURN_KEYS.covid,
            RETURN_KEYS.symptomatic,
            RETURN_KEYS.tested,
            RETURN_KEYS.secondary_infections,
            RETURN_KEYS.cases_prevented_social_distancing,
            RETURN_KEYS.cases_prevented_symptom_isolating,
            RETURN_KEYS.cases_prevented_contact_tracing,
            RETURN_KEYS.fractional_r,
        ],
        inplace=True,
    )

    return pd.concat({STATS_KEYS.mean: df.mean(0), STATS_KEYS.std: df.std(0)}, axis=1)


def scale_results(results, monte_carlo_factor, r_monte_carlo_factor, nppl):
    """
    Scales the scenario run results

    :param results: Original scenario run results
    :param monte_carlo_factor:
    :param r_monte_carlo_factor:
    :param nppl: Total population count to scale to
    :returns: Pandas dataframe with results scaled
    """
    results = results.copy()
    rvals = [RETURN_KEYS.base_r, RETURN_KEYS.reduced_r]

    scale = []
    for k in results.index:
        if k in rvals:
            scale.append(1)
        elif "%" in k:
            scale.append(100)
        else:
            scale.append(nppl)

    scale = pd.Series(scale, index=results.index)

    results[STATS_KEYS.mean] = results[STATS_KEYS.mean] * scale
    results[STATS_KEYS.std] = results[STATS_KEYS.std] * scale

    mc_scale = []
    for k in results.index:
        if k in rvals:
            mc_scale.append(r_monte_carlo_factor)
        elif "%" in k:
            mc_scale.append(1)
        else:
            mc_scale.append(monte_carlo_factor)

    mc_std_error_factors = pd.Series(mc_scale, index=results.index)

    results[STATS_KEYS.std] = results[STATS_KEYS.std] * mc_std_error_factors

    return results


def results_table(results_dct, index_name="scenario"):
    """
    Turn results represented as a dictionary into a pandas dataframe

    :param results_dct: Results dictionary
    :param index_name: Name of the dataframe index to use
    :returns: Results dataframe
    """
    df = pd.concat({k: v.T for k, v in results_dct.items()})
    df.index.names = [index_name, config.STATISTIC_COLNAME]
    return df
