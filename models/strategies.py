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

    if case.under_18:
        wfh = not do_schools_open
        met_before_w = met_before_s
    else:
        wfh = rng.uniform() < wfh_prob

    got_tested = rng.uniform() < policy_adherence

    # For speed pull the shape of these arrays once
    home_contacts = contacts['home_contacts']
    work_contacts = contacts['work_contacts']
    othr_contacts = contacts['othr_contacts']

    home_infections = contacts['home_infections']
    work_infections = contacts['work_infections']
    othr_infections = contacts['othr_infections']

    home_contacts_shape = contacts['home_contacts'].size
    work_contacts_shape = contacts['work_contacts'].size
    othr_contacts_shape = contacts['othr_contacts'].size

    # If policy to do app tracing
    if do_app_tracing:
        has_app = rng.uniform() < app_cov
        # If has app, we can trace contacts through the app. Assume home contacts not needed to trace this way
        if has_app:
            work_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=work_contacts_shape)
            othr_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=othr_contacts_shape)

    else:
        work_contacts_trace_app = np.zeros(shape=work_contacts_shape)
        othr_contacts_trace_app = np.zeros(shape=othr_contacts_shape)


    # If policy of manual tracing
    if do_manual_tracing:
        # Prob of manual tracing is a base chance, modified by the chance the person knows who the contact is.
        home_contacts_trace_manual = rng.binomial(n=1, p=manual_home_trace_prob * met_before_h, size=home_contacts_shape)
        work_contacts_trace_manual = rng.binomial(n=1, p=manual_work_trace_prob * met_before_w, size=work_contacts_shape)
        othr_contacts_trace_manual = rng.binomial(n=1, p=manual_othr_trace_prob * met_before_o, size=othr_contacts_shape)

    else:
        home_contacts_trace_manual = np.zeros(shape=home_contacts_shape)
        work_contacts_trace_manual = np.zeros(shape=work_contacts_shape)
        othr_contacts_trace_manual = np.zeros(shape=othr_contacts_shape)

    # TODO: Different from Kucharski
    # Work out if we have traced the individual in either way
    home_contacts_traced = home_contacts_trace_manual
    work_contacts_traced = work_contacts_trace_app | work_contacts_trace_manual
    othr_contacts_traced = othr_contacts_trace_app | othr_contacts_trace_manual

    # Work out if each contact will adhere to the policy
    home_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=home_contacts_shape)
    work_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=work_contacts_shape)
    othr_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=othr_contacts_shape)

    # Compute which contact will isolate because of the contact trace
    home_contacts_isolated = home_contacts_traced & home_contacts_adherence
    work_contacts_isolated = work_contacts_traced & work_contacts_adherence
    othr_contacts_isolated = othr_contacts_traced & othr_contacts_adherence

    # If the policy is to isolate as soon as you notice symptoms, work out which contacts will be prevented
    # TODO: assumes zero lag in the test -> result -> contact trace system
    if do_isolation:
        home_contacts_prevented = home_contacts >= case.noticed_symptoms
        work_contacts_prevented = work_contacts >= case.noticed_symptoms
        othr_contacts_prevented = othr_contacts >= case.noticed_symptoms

    else:
        home_contacts_prevented = np.zeros(shape=home_contacts_shape)
        work_contacts_prevented = np.zeros(shape=work_contacts_shape)
        othr_contacts_prevented = np.zeros(shape=othr_contacts_shape)

    ## Compute the base reproduction rate
    base_rr = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    ## Compute the reproduction rate due to the policy
    # Remove infections due to case isolation
    home_infections_post_policy = home_infections & (not home_contacts_prevented)
    work_infections_post_policy = work_infections & (not work_contacts_prevented)
    othr_infections_post_policy = othr_infections & (not othr_contacts_prevented)

    # Count traced contacts as not included in the R TODO: make a proportion
    home_infections_post_policy = home_infections_post_policy & home_contacts_isolated
    work_infections_post_policy = work_infections_post_policy & work_contacts_isolated
    othr_infections_post_policy = othr_infections_post_policy & othr_contacts_isolated

    reduced_rr = home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum()

    ## Count the number of manual traces needed

    manual_traces = home_contacts_trace_manual.sum() + work_contacts_trace_manual.sum() + othr_contacts_trace_manual.sum()

    return np.array([base_rr, reduced_rr, manual_traces])

