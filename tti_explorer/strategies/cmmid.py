import numpy as np

from . import registry
from .common import RETURN_KEYS


@registry("CMMID")
def CMMID_strategy(
    case,
    contacts,
    rng,
    do_isolation,
    do_manual_tracing,
    do_app_tracing,
    do_pop_testing,
    do_schools_open,
    manual_home_trace_prob,
    manual_work_trace_prob,
    manual_othr_trace_prob,
    met_before_w,
    met_before_s,
    met_before_o,
    max_contacts,
    wfh_prob,
    app_cov,
    p_pop_test,
    policy_adherence,
):
    """
    This is the implementation of the original Kucharski paper.

    It appears that the original code had a few typos which we replicate here to make sure we can reproduce original paper's results
    The typos are marked with comments below.

    The paper:
    Effectiveness of isolation, testing, contact tracing and physical distancing on reducing transmission of SARS-CoV-2 in different settings
    Kucharski AJ, Klepac P, Conlan AJK, Kissler S, Tang M et al. MedRxiv preprint, 2020.
    """

    if case.under18:
        wfh = not do_schools_open
        met_before_w = met_before_s
    else:
        wfh = rng.uniform() < wfh_prob

    # Get tested if symptomatic AND comply with policy
    symptomatic = case.symptomatic
    tested = rng.uniform() < policy_adherence
    # has_phone = rng.uniform() < app_cov

    # For speed pull the shape of these arrays once
    # home_contacts = contacts.home[:, 1]
    # work_contacts = contacts.work[:, 1]
    # othr_contacts = contacts.other[:, 1]

    home_infections = (contacts.home[:, 0] >= 0).astype(bool)
    work_infections = (contacts.work[:, 0] >= 0).astype(bool)
    othr_infections = (contacts.other[:, 0] >= 0).astype(bool)

    n_home = home_infections.shape[0]
    n_work = work_infections.shape[0]
    n_othr = othr_infections.shape[0]

    if tested and symptomatic and do_isolation:
        # Home contacts not necessarily contacted and infected on the same day
        inf_period = case.day_noticed_symptoms
    else:
        inf_period = 5.0

    pop_tested = rng.uniform() < p_pop_test
    if do_pop_testing and pop_tested:
        tested = True
        inf_period = rng.randint(6)

    if do_app_tracing:
        has_app = rng.uniform() < app_cov
        # If has app, we can trace contacts through the app. Assume home contacts not needed to trace this way
        if has_app:
            manual_work_trace_prob = app_cov
            manual_othr_trace_prob = app_cov
        else:
            manual_work_trace_prob = 0.0
            manual_othr_trace_prob = 0.0

    inf_ratio = inf_period / 5.0

    if wfh:
        inf_ratio_w = 0.0
    else:
        inf_ratio_w = inf_ratio

    if n_othr > 0:
        scale_othr = np.minimum(1.0, (max_contacts * 5.0) / n_othr)
    else:
        scale_othr = 1.0

    rr_basic_ii = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    home_infect = home_infections & rng.binomial(n=1, p=inf_ratio)
    work_infect = work_infections & rng.binomial(n=1, p=inf_ratio_w)
    othr_infect = othr_infections & rng.binomial(n=1, p=inf_ratio * scale_othr)
    rr_ii = home_infect.sum() + work_infect.sum() + othr_infect.sum()

    # home_traced = rng.binomial(n=1, p=manual_home_trace_prob, size=n_home).astype(bool)
    # work_traced = rng.binomial(
    #     n=1, p=manual_work_trace_prob * met_before_w, size=n_work
    # ).astype(bool)
    # # Typo from original paper: it should have been `manual_othr_trace_prob`
    # othr_traced = rng.binomial(
    #     n=1, p=manual_work_trace_prob * met_before_o * scale_othr, size=n_othr
    # ).astype(bool)

    home_averted = home_infect & rng.binomial(
        n=1, p=manual_home_trace_prob * policy_adherence, size=n_home
    ).astype(bool)
    work_averted = work_infect & rng.binomial(
        n=1, p=manual_work_trace_prob * met_before_w * policy_adherence, size=n_work
    ).astype(bool)
    # Typo from original paper: p here should be multiplied by `scale_othr`
    othr_averted = othr_infect & rng.binomial(
        n=1, p=met_before_o * manual_othr_trace_prob * policy_adherence, size=n_othr
    ).astype(bool)

    if tested & symptomatic & (do_manual_tracing | do_app_tracing):
        total_averted = home_averted.sum() + work_averted.sum() + othr_averted.sum()
    else:
        total_averted = 0.0

    rr_reduced = rr_ii - total_averted

    return {
        RETURN_KEYS.base_r: rr_basic_ii,
        RETURN_KEYS.reduced_r: rr_reduced,
        RETURN_KEYS.man_trace: 0,
    }
