import numpy as np

from utils import Registry

registry = Registry()

def limit_contact(contacts, max_per_day):
    """Generates a boolean array describing if a contact would not have 
    been contacted due daily contact limiting.

    Parameters
    ----------
    contacts : 1d array of contact days
    max_per_day : Max contacts per day
    """
    if contacts.size == 0:
        return np.array([]).astype(bool)
    contact_limited = np.zeros_like(contacts).astype(bool)
    for day in range(np.max(contacts)+1):
        is_day = (contacts == day)
        n_on_day = is_day.cumsum()
        allow_on_day = (n_on_day <= max_per_day) & (n_on_day != 0)
        contact_limited = (contact_limited | allow_on_day)

    return contact_limited

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

    # Get tested if symptomatic AND comply with policy 
    symptomatic = case.symptomatic
    tested = rng.uniform() < policy_adherence
    has_phone = rng.uniform() < app_cov

    # For speed pull the shape of these arrays once
    home_contacts = contacts.home[:, 1]
    work_contacts = contacts.work[:, 1]
    othr_contacts = contacts.other[:, 1]

    home_infections = (contacts.home[:, 0] >= 0).astype(bool)
    work_infections = (contacts.work[:, 0] >= 0).astype(bool)
    othr_infections = (contacts.other[:, 0] >= 0).astype(bool)

    n_home = home_infections.shape[0]
    n_work = work_infections.shape[0]
    n_othr = othr_infections.shape[0]

    # home_contacts_prevented = np.zeros_like(n_home).astype(bool)
    # work_contacts_prevented = np.zeros_like(n_work).astype(bool)
    # othr_contacts_prevented = np.zeros_like(n_othr).astype(bool)

    if tested and symptomatic and do_isolation:
        # Home contacts not necessarily contacted and infected on the same day
        # home_contacts_prevented = (contacts.home[:, 0] >= case.day_noticed_symptoms).astype(bool)
        # work_contacts_prevented = (work_contacts >= case.day_noticed_symptoms).astype(bool)
        # othr_contacts_prevented = (othr_contacts >= case.day_noticed_symptoms).astype(bool)
        inf_period = case.day_noticed_symptoms
    else:
        inf_period = 5.

    pop_tested = rng.uniform() < p_pop_test
    if do_pop_testing and pop_tested:
        tested = True
        # tested_day = rng.randint(6)
        inf_period = rng.randint(6)
        # home_contacts_prevented = (contacts.home[:, 0] >= tested_day).astype(bool)
        # work_contacts_prevented = (work_contacts >= tested_day).astype(bool)
        # othr_contacts_prevented = (othr_contacts >= tested_day).astype(bool)

    if do_app_tracing:
        has_app = rng.uniform() < app_cov
        # If has app, we can trace contacts through the app. Assume home contacts not needed to trace this way
        if has_app:
            manual_work_trace_prob = app_cov
            manual_othr_trace_prob = app_cov
        else:
            manual_work_trace_prob = 0.
            manual_othr_trace_prob = 0.

    inf_ratio = inf_period / 5.

    if wfh:
        inf_ratio_w = 0.
    else:
        inf_ratio_w = inf_ratio

    # othr_contacts_limited = limit_contact(othr_contacts, max_contacts)

    if n_othr > 0:
        scale_othr = np.minimum(1., (max_contacts * 5.) / n_othr)
    else:
        scale_othr = 1.0

    rr_basic_ii = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    home_infect = home_infections & rng.binomial(n=1, p=inf_ratio)
    work_infect = work_infections & rng.binomial(n=1, p=inf_ratio_w)
    othr_infect = othr_infections & rng.binomial(n=1, p=inf_ratio * scale_othr)
    rr_ii = home_infect.sum() + work_infect.sum() + othr_infect.sum()

    home_traced = rng.binomial(n=1, p=manual_home_trace_prob, size=n_home).astype(bool)
    work_traced = rng.binomial(n=1, p=manual_work_trace_prob * met_before_w, size=n_work).astype(bool)
    othr_traced = rng.binomial(n=1, p=manual_work_trace_prob * met_before_o * scale_othr, size=n_othr).astype(bool)
    
    home_averted = home_infect & rng.binomial(n=1, p=manual_home_trace_prob * policy_adherence, size=n_home).astype(bool)
    work_averted = work_infect & rng.binomial(n=1, p=manual_work_trace_prob * met_before_w * policy_adherence, size=n_work).astype(bool)
    othr_averted = othr_infect & rng.binomial(n=1, p=met_before_o * manual_othr_trace_prob * policy_adherence, size=n_othr).astype(bool)

    if tested & symptomatic & (do_manual_tracing | do_app_tracing):
        total_averted = home_averted.sum() + work_averted.sum() + othr_averted.sum()
    else:
        total_averted = 0.

    rr_reduced = rr_ii - total_averted

    return rr_basic_ii, rr_reduced, 0.
    # return home_infections.sum() / n_home, work_infections.sum() / n_work, othr_infections.sum() / n_othr, home_infections.sum() + work_infections.sum() + othr_infections.sum(), base_rr, reduced_rr, manual_traces

def CMMID_strategy_better(
    case, contacts, rng,

    do_individual_isolation,
    do_household_isolation,
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

    # Assume testing all symptomatic. tested if symptomatic AND comply with policy. TODO: Assumes no test lag
    got_symptom_tested = (rng.uniform() < policy_adherence) and case.symptomatic
    symptom_test_day = case.day_noticed_symptoms

    # get randomly tested if doing random testing
    got_random_tested = (rng.uniform() < p_pop_test) and do_pop_testing
    # Generate a random test day to be tested on. TODO: include incubation period
    random_test_day = rng.randint(6) # TODO: current infective period is 5 days.

    # For speed pull the shape of these arrays once
    home_contacts = contacts.home[:, 1]
    work_contacts = contacts.work[:, 1]
    othr_contacts = contacts.other[:, 1]

    home_infected_day = contacts.home[:, 0] # need to work out if isolating prevents an in home infection.
    home_infections = (contacts.home[:, 0] >= 0).astype(bool)
    work_infections = (contacts.work[:, 0] >= 0).astype(bool)
    othr_infections = (contacts.other[:, 0] >= 0).astype(bool)

    n_home = home_infections.shape[0]
    n_work = work_infections.shape[0]
    n_othr = othr_infections.shape[0]

    # If the person got tested
    if got_symptom_tested or got_random_tested:
        ### TRACING
        # If policy to do app tracing
        if do_app_tracing:
            has_app = rng.uniform() < app_cov
            # If has app, we can trace contacts through the app. Assume home contacts not needed to trace this way
            if has_app:
                # Trace contacts based on app coverage
                work_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=n_work).astype(bool)
                othr_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=n_othr).astype(bool)
            else:
                work_contacts_trace_app = np.zeros(shape=n_work, dtype=bool)
                othr_contacts_trace_app = np.zeros(shape=n_othr, dtype=bool)
        else:
            work_contacts_trace_app = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_trace_app = np.zeros(shape=n_othr, dtype=bool)

        # If policy of manual tracing
        if do_manual_tracing:
            # Prob of manual tracing is a base chance, modified by the chance the person knows who the contact is.
            # home_contacts_trace_manual = rng.binomial(n=1, p=manual_home_trace_prob * met_before_h, size=n_home).astype(bool)
            work_contacts_trace_manual = rng.binomial(n=1, p=manual_work_trace_prob * met_before_w, size=n_work).astype(bool)
            othr_contacts_trace_manual = rng.binomial(n=1, p=manual_othr_trace_prob * met_before_o, size=n_othr).astype(bool)
        else:
            # home_contacts_trace_manual = np.zeros(shape=n_home, dtype=bool)
            work_contacts_trace_manual = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_trace_manual = np.zeros(shape=n_othr, dtype=bool)

        # TODO: Different from Kucharski
        # If we have the household isolation policy, (TODO: makes no sense without the other tracing policies)
        # Isolate the household. TODO: Doesn't distinguish between prevented infections and traced
        if do_household_isolation:
            home_contacts_trace = np.ones(shape=n_home, dtype=bool)
        else:
            home_contacts_trace = np.zeros(shape=n_home, dtype=bool)

        # Traced if traced either way
        work_contacts_traced = work_contacts_trace_app | work_contacts_trace_manual
        othr_contacts_traced = othr_contacts_trace_app | othr_contacts_trace_manual

        # Compute trace statistics
        manual_traces = work_contacts_trace_manual.sum() + othr_contacts_trace_manual.sum()
        app_traces = work_contacts_trace_app.sum() + othr_contacts_trace_app.sum()

        # Work out if each contact will adhere to the policy
        home_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=n_home).astype(bool)
        work_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=n_work).astype(bool)
        othr_contacts_adherence = rng.binomial(n=1, p=policy_adherence, size=n_othr).astype(bool)

        # Compute which contact will isolate because of the contact trace
        # TODO: Assumes traced contact does not count towards R
        home_contacts_isolated = home_contacts_traced & home_contacts_adherence 
        work_contacts_isolated = work_contacts_traced & work_contacts_adherence
        othr_contacts_isolated = othr_contacts_traced & othr_contacts_adherence

        ### ISOLATING

        # Find the day isolated on to determine when removed from contact.
        # TODO: Simplify this logic.
        if got_symptom_tested and got_random_tested:
            isolate_day = np.minimum(symptom_test_day, random_test_day)
        elif got_symptom_tested:
            isolate_day = symptom_test_day
        elif got_random_tested:
            isolate_day = random_test_day

        # Prevent contact after isolation day
        home_contacts_prevented = (home_infections >= isolate_day).astype(bool)
        work_contacts_prevented = (work_contacts >= isolate_day).astype(bool)
        othr_contacts_prevented = (othr_contacts >= isolate_day).astype(bool)
    else:
        # No tracing took place if they didn't get tested positive.
        home_contacts_traced = np.zeros(shape=n_home, dtype=bool)
        work_contacts_traced = np.zeros(shape=n_work, dtype=bool)
        othr_contacts_traced = np.zeros(shape=n_othr, dtype=bool)

        # Default cases prevented (none)
        home_contacts_prevented = np.zeros(shape=n_home, dtype=bool)
        work_contacts_prevented = np.zeros(shape=n_work, dtype=bool)
        othr_contacts_prevented = np.zeros(shape=n_othr, dtype=bool)

        manual_traces = 0
        app_traces = 0

    # Compute reduction in contacts due to contact limiting policy. Independent of test status.
    othr_contacts_limited = limit_contact(othr_contacts, max_contacts)

    # Compute reduction in contacts due to wfh. Independent of test status.
    if wfh:
        work_contacts_wfh_limited = np.zeros_like(work_contacts).astype(bool)
    else:
        work_contacts_wfh_limited = np.ones_like(work_contacts).astype(bool)

    ## Compute the base reproduction rate
    base_rr = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    ## Compute the reproduction rate due to the policy
    # Remove infections due to case isolation
    home_infections_post_policy = home_infections & ~( home_contacts_prevented)
    work_infections_post_policy = work_infections & ~( work_contacts_prevented)
    othr_infections_post_policy = othr_infections & ~( othr_contacts_prevented)

    # Count traced contacts as not included in the R TODO: make a proportion
    home_infections_post_policy = home_infections_post_policy & ~home_contacts_isolated
    work_infections_post_policy = work_infections_post_policy & ~work_contacts_isolated
    othr_infections_post_policy = othr_infections_post_policy & ~othr_contacts_isolated

    # Remove contacts not made due to work from home
    work_infections_post_policy = work_infections_post_policy & work_contacts_wfh_limited

    # Remove other contact limiting contacts
    othr_infections_post_policy = othr_infections_post_policy & othr_contacts_limited

    # Count the reduced infection rate
    reduced_rr = home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum()

    return base_rr, reduced_rr, manual_traces

