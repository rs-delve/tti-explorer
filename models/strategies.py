import numpy as np

from utils import Registry

registry = Registry()

@registry("CMMID")
def CMMID_strategy(
    case, contacts, rng,

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
    met_before_h,
    met_before_o,
    max_contacts,

    wfh_prob,
    app_cov,
    p_pop_test,
    policy_adherence
):

    if case.under18:
        wfh = not do_schools_open
        met_before_w = met_before_s
    else:
        wfh = rng.uniform() < wfh_prob

    got_tested = rng.uniform() < policy_adherence

    # For speed pull the shape of these arrays once
    home_contacts = contacts.home[:, 1]
    work_contacts = contacts.work[:, 1]
    othr_contacts = contacts.other[:, 1]

    home_infections = contacts.home[:, 0] >= 0
    work_infections = contacts.work[:, 0] >= 0
    othr_infections = contacts.other[:, 0] >= 0

    n_home = home_infections.shape[0]
    n_work = work_infections.shape[0]
    n_othr = othr_infections.shape[0]

    # If policy to do app tracing
    if do_app_tracing:
        has_app = rng.uniform() < app_cov
        # If has app, we can trace contacts through the app. Assume home contacts not needed to trace this way
        if has_app:
            work_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=n_work)
            othr_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=n_othr)

    else:
        work_contacts_trace_app = np.zeros(shape=n_work, dtype=int)
        othr_contacts_trace_app = np.zeros(shape=n_othr, dtype=int)


    # If policy of manual tracing
    if do_manual_tracing:
        # Prob of manual tracing is a base chance, modified by the chance the person knows who the contact is.
        home_contacts_trace_manual = rng.binomial(n=1, p=manual_home_trace_prob * met_before_h, size=n_home)
        work_contacts_trace_manual = rng.binomial(n=1, p=manual_work_trace_prob * met_before_w, size=n_work)
        othr_contacts_trace_manual = rng.binomial(n=1, p=manual_othr_trace_prob * met_before_o, size=n_othr)

    else:
        home_contacts_trace_manual = np.zeros(shape=n_home, dtype=int)
        work_contacts_trace_manual = np.zeros(shape=n_work, dtype=int)
        othr_contacts_trace_manual = np.zeros(shape=n_othr, dtype=int)

    # TODO: Different from Kucharski
    # Work out if we have traced the individual in either way
    home_contacts_traced = home_contacts_trace_manual
    work_contacts_traced = work_contacts_trace_app | work_contacts_trace_manual
    othr_contacts_traced = othr_contacts_trace_app | othr_contacts_trace_manual

    # Work out if each contact will adhere to the policy
    home_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=n_home)
    work_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=n_work)
    othr_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=n_othr)

    # Compute which contact will isolate because of the contact trace
    home_contacts_isolated = home_contacts_traced & home_contacts_adherence
    work_contacts_isolated = work_contacts_traced & work_contacts_adherence
    othr_contacts_isolated = othr_contacts_traced & othr_contacts_adherence

    # If the policy is to isolate as soon as you notice symptoms, work out which contacts will be prevented
    # TODO: assumes zero lag in the test -> result -> contact trace system
    if do_isolation:
    # BE: is this calculation correct? shouldn't prevention be conditional on them having covid too? (not just symptoms)
        home_contacts_prevented = home_contacts >= case.day_noticed_symptoms
        work_contacts_prevented = work_contacts >= case.day_noticed_symptoms
        othr_contacts_prevented = othr_contacts >= case.day_noticed_symptoms

    else:
        home_contacts_prevented = np.zeros(shape=n_home, dtype=int)
        work_contacts_prevented = np.zeros(shape=n_work, dtype=int)
        othr_contacts_prevented = np.zeros(shape=n_othr, dtype=int)

    ## Compute the base reproduction rate
    base_rr = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    ## Compute the reproduction rate due to the policy
    # Remove infections due to case isolation
    home_infections_post_policy = home_infections & (~ home_contacts_prevented)
    work_infections_post_policy = work_infections & (~ work_contacts_prevented)
    othr_infections_post_policy = othr_infections & (~ othr_contacts_prevented)

    # Count traced contacts as not included in the R TODO: make a proportion
    home_infections_post_policy = home_infections_post_policy & home_contacts_isolated
    work_infections_post_policy = work_infections_post_policy & work_contacts_isolated
    othr_infections_post_policy = othr_infections_post_policy & othr_contacts_isolated

    reduced_rr = home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum()

    ## Count the number of manual traces needed

    manual_traces = home_contacts_trace_manual.sum() + work_contacts_trace_manual.sum() + othr_contacts_trace_manual.sum()

    return base_rr, reduced_rr, manual_traces

