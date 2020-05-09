from types import SimpleNamespace

import numpy as np

from utils import Registry

registry = Registry()

# Changing the string values here saves having to change them
# in every strategy and will keep tables consistent
RETURN_KEYS = SimpleNamespace(
        base_r="Base R",
        reduced_r='Reduced R',
        man_trace='Manual Traces',
        app_trace='App Traces',
        tests='Tests Needed',
        quarantine='PersonDays Quarantined',
        wasted_quarantine='Wasted PersonDays Quarantined'
    )

# BE: this type of masking might be useful to limit contacts
# for home contacts n_days would be 1
def limit_contact_mask(n_daily, n_days, max_per_day):
    return np.repeat(np.arange(1, n_daily + 1), n_days) <= max_per_day


def limit_contact(contacts, max_per_day):
    """Generates a boolean array describing if a contact would have 
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

    return {
            RETURN_KEYS.base_r: rr_basic_ii,
            RETURN_KEYS.reduced_r: rr_reduced,
            RETURN_KEYS.man_trace: 0
        }
    # return home_infections.sum() / n_home, work_infections.sum() / n_work, othr_infections.sum() / n_othr, home_infections.sum() + work_infections.sum() + othr_infections.sum(), base_rr, reduced_rr, manual_traces

@registry("CMMID_better")
def CMMID_strategy_better(
    case, contacts, rng,

    do_individual_isolation,
    do_household_isolation,

    do_manual_tracing,
    do_app_tracing,

    do_pop_testing,
    do_symptom_testing,

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
    got_symptom_tested = (rng.uniform() < policy_adherence) and case.symptomatic and do_symptom_testing
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
            home_contacts_traced = np.ones(shape=n_home, dtype=bool)
        else:
            home_contacts_traced = np.zeros(shape=n_home, dtype=bool)

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
        if do_individual_isolation:
            home_contacts_prevented = (home_infections >= isolate_day).astype(bool)
            work_contacts_prevented = (work_contacts >= isolate_day).astype(bool)
            othr_contacts_prevented = (othr_contacts >= isolate_day).astype(bool)
        else:
            home_contacts_prevented = np.zeros(shape=n_home, dtype=bool)
            work_contacts_prevented = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_prevented = np.zeros(shape=n_othr, dtype=bool)
    else:
        # No tracing took place if they didn't get tested positive.
        home_contacts_isolated = np.zeros(shape=n_home, dtype=bool)
        work_contacts_isolated = np.zeros(shape=n_work, dtype=bool)
        othr_contacts_isolated = np.zeros(shape=n_othr, dtype=bool)

        # Default cases prevented (none)
        home_contacts_prevented = np.zeros(shape=n_home, dtype=bool)
        work_contacts_prevented = np.zeros(shape=n_work, dtype=bool)
        othr_contacts_prevented = np.zeros(shape=n_othr, dtype=bool)

        manual_traces = 0
        app_traces = 0

    # Compute reduction in contacts due to contact limiting policy. Independent of test status.
    othr_contacts_limited = ~limit_contact(othr_contacts, max_contacts)

    # Compute reduction in contacts due to wfh. Independent of test status.
    if wfh:
        work_contacts_wfh_limited = np.ones_like(work_contacts).astype(bool)
    else:
        work_contacts_wfh_limited = np.zeros_like(work_contacts).astype(bool)

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
    work_infections_post_policy = work_infections_post_policy & ~work_contacts_wfh_limited

    # Remove other contact limiting contacts
    othr_infections_post_policy = othr_infections_post_policy & ~othr_contacts_limited

    # Count the reduced infection rate
    reduced_rr = home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum()

    return {
            RETURN_KEYS.base_r: base_rr,
            RETURN_KEYS.reduced_r: reduced_rr,
            RETURN_KEYS.man_trace: manual_traces
        }


@registry("temporal_anne_flowchart")
def temporal_anne_flowchart(
    case, contacts, rng,

    # proportion_symptomatic,         # The proportion of positive cases that are symptomatic

    isolate_individual_on_symptoms, # Isolate the individual after they present with symptoms
    isolate_individual_on_positive, # Isolate the individual after they test positive

    isolate_household_on_symptoms,  # Isolate the household after individual present with symptoms
    isolate_household_on_positive,  # Isolate the household after individual test positive

    isolate_contacts_on_symptoms,   # Isolate the contacts after individual present with symptoms
    isolate_contacts_on_positive,   # Isolate the contacts after individual test positive

    # test_contacts_on_positive,    # TODO: this option will require more work  # Do we test contacts of a positive case immediately, or wait for them to develop symptoms

    do_symptom_testing,             # Test symptomatic individuals
    app_cov,                        # % Coverage of the app
    app_report_prob,                # Likelihood of reporting symptoms through app
    manual_report_prob,             # Likelihood of manually reporting symptoms (will also do if has app but didn't report through it. See flowchart)

    testing_delay,                  # Delay between test and results

    do_manual_tracing,              # Perform manual tracing of contacts
    do_app_tracing,                 # Perform app tracing of contacts

    manual_home_trace_prob,         # Probability of manually tracing a home contact
    manual_work_trace_prob,         # Probability of manually tracing a work contact
    manual_othr_trace_prob,         # Probability of manually tracing an other contact

    trace_adherence,                # Probability of a traced contact isolating correctly

    do_schools_open,                # If schools are open

    met_before_h,                   # Probability of having met a home contact before to be able to manually trace
    met_before_w,                   # Probability of having met a work contact before to be able to manually trace
    met_before_s,                   # Probability of having met a school contact before to be able to manually trace
    met_before_o,                   # Probability of having met a other contact before to be able to manually trace

    max_contacts,                   # Place a limit on the number of other contacts per day

    wfh_prob,                       # Proportion or the population working from home

    fractional_infections,          # Include infected but traced individuals as a fraction of their infection period not isolated

    quarantine_length,              # Length of quarantine imposed on COVID cases (and household)

    latent_period,              # Length of a cases incubation period (from infection to start of infectious period) 

    ## Returns: 
    #   - base_rr : number of infections caused without any measures. NaN if doesn't have COVID.
    #   - reduced_rr : number of infections caused with measures. To include fractional cases if desired.
    #   - n_manual_traces : number of manual traces this case causes
    #   - n_app_traces : number of app traces caused. Not exclusive to manual traces
    #   - n_tests_performed : number of tests this cause caused to happen (i.e. in contacts + self).
    #   - person_days_quarantine : number of days all individuals affected spend in quarantine
    #   - person_days_wasted_quarantine : number of days individuals without COVID were locked down
):

    # If under 18, change wfh and likelihood of knowing contacts
    if case.under18:
        wfh = not do_schools_open
        met_before_w = met_before_s
    else:
        wfh = rng.uniform() < wfh_prob

    # Test if user has the app
    has_app = rng.uniform() < app_cov

    # If the case is symptomatic, test if case reports and through what channel. Assume reports on day noticed
    if case.symptomatic:
        if has_app:
            # report through app probability
            if rng.uniform() < app_report_prob:
                report_app = True
                report_manual = False
            # if doesn't report through app, may still report manually
            elif rng.uniform() < manual_report_prob:
                report_app = False
                report_manual = True
            # Will not report otherwise
            else:
                report_app = False
                report_manual = False
        else:
            # If doesn't have app, may report manually
            if rng.uniform() < manual_report_prob:
                report_app = False
                report_manual = True
            # Will not report otherwise
            else:
                report_app = False
                report_manual = False
    # Will not report is non symptomatic.
    else:
        report_app = False
        report_manual = False

    # Check if any test was performed
    test_performed = (report_app or report_manual) and do_symptom_testing
    if test_performed:
        test_perform_day = case.day_noticed_symptoms
        test_results_day = test_perform_day + testing_delay

    # Days on which individual made contact with their contacts. For home, earliest day of infectivity.
    home_contacts = contacts.home[:, 1]
    work_contacts = contacts.work[:, 1]
    othr_contacts = contacts.other[:, 1]

    # Get the day on which a household member was infected
    home_infected_day = contacts.home[:, 0]
    # Get if an infection was caused in contacts
    home_infections = (contacts.home[:, 0] >= 0).astype(bool)
    work_infections = (contacts.work[:, 0] >= 0).astype(bool)
    othr_infections = (contacts.other[:, 0] >= 0).astype(bool)

    # Pre pull numbers of contacts for speed
    n_home = home_infections.shape[0]
    n_work = work_infections.shape[0]
    n_othr = othr_infections.shape[0]


    # Compute reduction in contacts due to contact limiting policy. Independent of test status.
    othr_contacts_limited = ~limit_contact(othr_contacts, max_contacts)

    # Compute reduction in contacts due to wfh. Independent of test status.
    if wfh:
        work_contacts_wfh_limited = np.ones_like(work_contacts).astype(bool)
    else:
        work_contacts_wfh_limited = np.zeros_like(work_contacts).astype(bool)


    # If the person got tested
    if test_performed:

        ### ISOLATING

        # If isolate the person on day of test
        if isolate_individual_on_symptoms:
            isolate_day = test_perform_day
        # If isolate on positive and would be positive
        elif isolate_individual_on_positive and case.covid:
            isolate_day = test_results_day
        else:
            # TODO: Should never get here. Nan > all numbers but will cause warning.
            isolate_day = np.nan

        # Prevent contact after isolation day
        home_contacts_prevented = (home_infected_day >= isolate_day).astype(bool)
        work_contacts_prevented = (work_contacts >= isolate_day).astype(bool)
        othr_contacts_prevented = (othr_contacts >= isolate_day).astype(bool)

        # Remove contacts not made due to work from home
        work_contacts_prevented = work_contacts_prevented | work_contacts_wfh_limited

        # Remove other contact limiting contacts
        othr_contacts_prevented = othr_contacts_prevented | othr_contacts_limited

        ### TRACING CONTACTS

        # If policy to do app tracing
        if do_app_tracing:
            # If has app and reported through it, we can trace contacts through the app. Assume home contacts not needed to trace this way
            # TODO: can we trace through the app those who have it, but didn't report through it?
            if report_app:
                # Trace contacts based on app coverage
                work_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=n_work).astype(bool)
                othr_contacts_trace_app = rng.binomial(n=1, p=app_cov, size=n_othr).astype(bool)
            else:
                # No app contacts traced
                work_contacts_trace_app = np.zeros(shape=n_work, dtype=bool)
                othr_contacts_trace_app = np.zeros(shape=n_othr, dtype=bool)
        else:
            work_contacts_trace_app = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_trace_app = np.zeros(shape=n_othr, dtype=bool)

        # If policy of manual tracing
        if do_manual_tracing:
            # Prob of manual tracing is a base chance, modified by the chance the person knows who the contact is.
            work_contacts_trace_manual = rng.binomial(n=1, p=manual_work_trace_prob * met_before_w, size=n_work).astype(bool)
            othr_contacts_trace_manual = rng.binomial(n=1, p=manual_othr_trace_prob * met_before_o, size=n_othr).astype(bool)
        else:
            # no contacts manually traced.
            work_contacts_trace_manual = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_trace_manual = np.zeros(shape=n_othr, dtype=bool)

        # Assume all home contacts traced 
        home_contacts_traced = np.ones_like(n_home, dtype=bool)

        # Remove traces that didn't happen as case was isolated
        work_contacts_trace_app = work_contacts_trace_app & ~work_contacts_prevented
        othr_contacts_trace_app = othr_contacts_trace_app & ~othr_contacts_prevented

        # Remove traces that didn't happen as case was isolated
        work_contacts_trace_manual = work_contacts_trace_manual & ~work_contacts_prevented
        othr_contacts_trace_manual = othr_contacts_trace_manual & ~othr_contacts_prevented
        
        # Traced if traced either way and didn't isolate and prevent contact
        work_contacts_traced = (work_contacts_trace_app | work_contacts_trace_manual)
        othr_contacts_traced = (othr_contacts_trace_app | othr_contacts_trace_manual)

        # Compute trace statistics
        # Only trace if we want to isolate
        if isolate_contacts_on_symptoms or (isolate_contacts_on_positive and case.covid):
            manual_traces = work_contacts_trace_manual.sum() + othr_contacts_trace_manual.sum()
            app_traces = work_contacts_trace_app.sum() + othr_contacts_trace_app.sum()
        else:
            manual_traces = 0.
            app_traces = 0.

        # Work out if each contact will adhere to the policy
        home_contacts_adherence = rng.binomial(n=1, p=trace_adherence, size=n_home).astype(bool)
        work_contacts_adherence = rng.binomial(n=1, p=trace_adherence, size=n_work).astype(bool)
        othr_contacts_adherence = rng.binomial(n=1, p=trace_adherence, size=n_othr).astype(bool)

        # Compute which contact will isolate because of the contact trace
        home_contacts_isolated = home_contacts_traced & home_contacts_adherence & ~home_contacts_prevented 
        work_contacts_isolated = work_contacts_traced & work_contacts_adherence & ~work_contacts_prevented
        othr_contacts_isolated = othr_contacts_traced & othr_contacts_adherence & ~othr_contacts_prevented


        ## Compute tests required
        # Assume we test every contact that is isolated
        # Those who would be quarantined are those who are isolating, but not totally prevented (due to the way isolated is computed)
        # home_contacts_quarantined = home_contacts_isolated 
        # work_contacts_quarantined = work_contacts_isolated 
        # othr_contacts_quarantined = othr_contacts_isolated 

        # count own test
        total_tests_performed = 1
        # If house isolated on symptoms, or on positive
        # TODO: Assume for now that we only TEST contacts AFTER the primary tests positive
        # TODO: After discussion, we will not test home contacts until they develop symptoms. 
        # These tests will not count against the primary case, as these would have been tested regardless.
        if case.covid:
            total_tests_performed += 0. # home_contacts_isolated.sum()

        # If contacts isolated on symptoms, or on positive
        # TODO: Again, after conversations, we will not test traced contacts unless a particular policy decision is made.
        # We do not count cases that would become positive and symptomatic against the primary case, but do count others. 
        if case.covid: # and test_contacts_on_positive:
            total_tests_performed += 0 # work_contacts_isolated.sum() + othr_contacts_isolated.sum()

        ## Compute the quarantine days

        person_days_quarantine = 0
        person_days_wasted_quarantine = 0

        # If person has covid, require full lockdown
        if case.covid and (isolate_individual_on_symptoms or isolate_individual_on_positive):
            person_days_quarantine += quarantine_length
        # If not, only require the test delay of quarantine. These days are "wasted"
        elif isolate_individual_on_symptoms:      
            person_days_quarantine += testing_delay
            person_days_wasted_quarantine += testing_delay
        ## Don't add any if: not isolating at all, individual only isolating after test complete

        # For household contacts, if isolating on symptoms and not covid, waste test delay days 
        if isolate_household_on_symptoms and not case.covid:
            # TODO: only counts home contacts that actually isolate
            person_days_quarantine += testing_delay * home_contacts_isolated.sum()
            person_days_wasted_quarantine += testing_delay * home_contacts_isolated.sum()
        # If person has covid, whole house will have to do full lockdown for the period
        # TODO: might be able to let some out if the test negative?
        elif (isolate_household_on_positive or isolate_household_on_symptoms) and case.covid:
            person_days_quarantine += quarantine_length * home_contacts_isolated.sum()
            # TODO: Count as wasted the time that house members who do not have covid locked down as wasted
            person_days_wasted_quarantine += quarantine_length * (home_contacts_isolated & ~home_infections).sum()
        ## Don't add any if: Not isolating at all, or if waiting for positive test to isolate and doesn't have coivd

        # For traced contacts, if isolating on positive and doesn't have covid, waste test days
        ## NOTE: working with "quarantined" as this represents those traced who were still contacted and complied
        if isolate_contacts_on_symptoms and not case.covid:
            person_days_quarantine += testing_delay * (work_contacts_isolated.sum() + othr_contacts_isolated.sum())
            person_days_wasted_quarantine += testing_delay * (work_contacts_isolated.sum() + othr_contacts_isolated.sum())
        elif (isolate_contacts_on_positive or isolate_contacts_on_symptoms) and case.covid:
            # NOTE: for now assume that people are tested on the same day as isolated. So for contacts, same day as
            # the primary case if isolating on the day of symptoms, 3 days later if delaying. Will be the same number of days regardless.
            #  Probably would be an additional lag here.
                       
            # All quarantined for 3 days at least to get test
            person_days_quarantine += testing_delay * (work_contacts_isolated.sum() + othr_contacts_isolated.sum())
            # Those testing positive will be fully quarantined (minus the test lag days counted)
            person_days_quarantine += (quarantine_length - testing_delay) * ((work_contacts_isolated & work_infections).sum() + (othr_contacts_isolated & othr_infections).sum())
            # Those who test negative will have "wasted" the 3 days
            person_days_wasted_quarantine += (testing_delay) * ((work_contacts_isolated & ~work_infections).sum() + (othr_contacts_isolated & ~othr_infections).sum())

    else:
        # No tracing took place if they didn't get tested positive.
        home_contacts_isolated = np.zeros(shape=n_home, dtype=bool)
        work_contacts_isolated = np.zeros(shape=n_work, dtype=bool)
        othr_contacts_isolated = np.zeros(shape=n_othr, dtype=bool)

        # Default cases prevented (none)
        home_contacts_prevented = np.zeros(shape=n_home, dtype=bool)
        work_contacts_prevented = work_contacts_wfh_limited
        othr_contacts_prevented = othr_contacts_limited

        manual_traces = 0
        app_traces = 0
        total_tests_performed = 0

        person_days_quarantine = 0
        person_days_wasted_quarantine = 0


    ## Compute the base reproduction rate
    base_rr = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    ## Compute the reproduction rate due to the policy
    # Remove infections due to case isolation
    home_infections_post_policy = home_infections & ~home_contacts_prevented
    work_infections_post_policy = work_infections & ~work_contacts_prevented
    othr_infections_post_policy = othr_infections & ~othr_contacts_prevented

    # Count traced contacts as not included in the R TODO: make a proportion
    home_infections_post_policy = home_infections_post_policy & ~home_contacts_isolated
    work_infections_post_policy = work_infections_post_policy & ~work_contacts_isolated
    othr_infections_post_policy = othr_infections_post_policy & ~othr_contacts_isolated
        
    ## Count fractional cases - will only occur if got tested
    if test_performed and fractional_infections:
        infectiousness_by_day = case.inf_profile
        cumulative_infectiousness = np.cumsum(infectiousness_by_day)
        infectious_period = len(infectiousness_by_day)

        # Get the days on which infections that were quarantined happened
        home_infection_days = home_infected_day[home_infections & ~home_infections_post_policy]
        work_infection_days = work_contacts[work_infections & ~work_infections_post_policy]
        othr_infection_days = othr_contacts[othr_infections & ~othr_infections_post_policy]

        # Compute day of contact becoming infectious after case started being infectious
        home_infectious_start = home_infection_days + latent_period
        work_infectious_start = work_infection_days + latent_period
        othr_infectious_start = othr_infection_days + latent_period

        # Compute the days home cases are left out in the world infectious
        if isolate_household_on_symptoms:
            home_infections_days_not_quarantined = test_perform_day - home_infectious_start
        elif isolate_household_on_positive:
            home_infections_days_not_quarantined = test_results_day - home_infectious_start
        else:
            # If neither of these are true, then the case would not have made it to here as would have been in hom_infections_post_policy
            home_infections_days_not_quarantined = (len(home_infectious_start)) * np.ones(n_home, dtype=int)

        # Compute the days a work/othr case is left out in the world infectious
        if isolate_contacts_on_symptoms:
            work_infections_days_not_quarantined = test_perform_day - work_infectious_start
            othr_infections_days_not_quarantined = test_perform_day - othr_infectious_start
        elif isolate_contacts_on_positive:
            work_infections_days_not_quarantined = test_results_day - work_infectious_start
            othr_infections_days_not_quarantined = test_results_day - othr_infectious_start
        else:
            work_infections_days_not_quarantined = (len(cumulative_infectiousness)) * np.ones(len(work_infectious_start), dtype=int)
            othr_infections_days_not_quarantined = (len(cumulative_infectiousness)) * np.ones(len(othr_infectious_start), dtype=int)

        # Only care about ones where there is more than zero days spent unisolated
        home_infections_days_not_quarantined = home_infections_days_not_quarantined[home_infections_days_not_quarantined > 0]
        work_infections_days_not_quarantined = work_infections_days_not_quarantined[work_infections_days_not_quarantined > 0]
        othr_infections_days_not_quarantined = othr_infections_days_not_quarantined[othr_infections_days_not_quarantined > 0]

        # Add one to get indexing correct - 1st day infectious is 0 in array
        home_cumulative_infectiousness = cumulative_infectiousness[home_infections_days_not_quarantined - 1].sum()
        work_cumulative_infectiousness = cumulative_infectiousness[work_infections_days_not_quarantined - 1].sum()
        othr_cumulative_infectiousness = cumulative_infectiousness[othr_infections_days_not_quarantined - 1].sum()

        fractional_R = home_cumulative_infectiousness + work_cumulative_infectiousness + othr_cumulative_infectiousness
    else:
        fractional_R = 0.
        

    # Count the reduced infection rate
    reduced_rr = home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum() + fractional_R

    return {
            RETURN_KEYS.base_r: base_rr if case.covid else np.nan,
            RETURN_KEYS.reduced_r: reduced_rr if case.covid else np.nan,
            RETURN_KEYS.man_trace: manual_traces,
            RETURN_KEYS.app_trace: app_traces,
            RETURN_KEYS.tests: total_tests_performed,
            RETURN_KEYS.quarantine: person_days_quarantine,
            RETURN_KEYS.wasted_quarantine: person_days_wasted_quarantine
        }
