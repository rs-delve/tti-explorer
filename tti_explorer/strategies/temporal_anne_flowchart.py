import numpy as np

from .. import config
from . import registry
from .common import _limit_contact, RETURN_KEYS


@registry('temporal_anne_flowchart')
def temporal_anne_flowchart(
    case, contacts, rng,

    # proportion_symptomatic,         # The proportion of positive cases that are symptomatic

    isolate_individual_on_symptoms, # Isolate the individual after they present with symptoms
    isolate_individual_on_positive, # Isolate the individual after they test positive

    isolate_household_on_symptoms,  # Isolate the household after individual present with symptoms
    isolate_household_on_positive,  # Isolate the household after individual test positive

    isolate_contacts_on_symptoms,   # Isolate the contacts after individual present with symptoms
    isolate_contacts_on_positive,   # Isolate the contacts after individual test positive

    test_contacts_on_positive,      # Do we test contacts of a positive case immediately, or wait for them to develop symptoms

    do_symptom_testing,             # Test symptomatic individuals
    app_cov,                        # % Coverage of the app

    testing_delay,                  # Delay between test and results

    do_manual_tracing,              # Perform manual tracing of contacts
    do_app_tracing,                 # Perform app tracing of contacts

    app_trace_delay,                # Delay associated with tracing through the app
    manual_trace_delay,             # Delay associated with tracing manually

    manual_home_trace_prob,         # Probability of manually tracing a home contact
    manual_work_trace_prob,         # Probability of manually tracing a work contact
    manual_othr_trace_prob,         # Probability of manually tracing an other contact

    trace_adherence,                # Probability of a traced contact isolating correctly

    go_to_school_prob,                # If schools are open

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
):
    """
    This is an implementation of flowchart produced by Anne Johnson and Guy Harling
    """

    # If under 18, change wfh and likelihood of knowing contacts
    if case.under18:
        wfh = rng.uniform() < 1 - go_to_school_prob
        met_before_w = met_before_s
    else:
        wfh = rng.uniform() < wfh_prob

    # Test if user has the app
    has_app = rng.uniform() < app_cov

    # If the case is symptomatic, test if case reports and through what channel. Assume reports on day noticed
    # TODO: logic changed. Assume if have app, this is how they will report.
    if case.symptomatic:
        does_report = rng.uniform() < trace_adherence

        if does_report:
            if has_app:
                report_app = True
                report_manual = False
            else:
                report_app = False
                report_manual = True
        else:
            report_app = False
            report_manual = False
    else:
        report_app = False
        report_manual = False

    # Check if any test was performed
    test_performed = (report_app or report_manual)
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
    othr_contacts_limited = ~_limit_contact(othr_contacts, max_contacts)

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
            # Don't isolate, set to something beyond simulation horizon
            isolate_day = 200

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
            if report_app and (isolate_contacts_on_symptoms or (isolate_contacts_on_positive and case.covid)):
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
        if do_manual_tracing and (isolate_contacts_on_symptoms or (isolate_contacts_on_positive and case.covid)):
            # Prob of manual tracing is a base chance, modified by the chance the person knows who the contact is.
            work_contacts_trace_manual = rng.binomial(n=1, p=manual_work_trace_prob * met_before_w, size=n_work).astype(bool)
            othr_contacts_trace_manual = rng.binomial(n=1, p=manual_othr_trace_prob * met_before_o, size=n_othr).astype(bool)
        else:
            # no contacts manually traced.
            work_contacts_trace_manual = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_trace_manual = np.zeros(shape=n_othr, dtype=bool)

        # Assume all home contacts traced
        if isolate_household_on_symptoms or (isolate_household_on_positive and case.covid):
            home_contacts_traced = np.ones_like(n_home, dtype=bool)
        else:
            home_contacts_traced = np.zeros(shape=n_home, dtype=bool)

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

        # Do tests on the positive contacts if we want to, and find out which are asymptomatic 
        if test_contacts_on_positive:
            # Cases that would be symptomatic
            work_symptomatic = rng.binomial(n=1, p=config.PROP_COVID_SYMPTOMATIC, size=n_work).astype(bool) & work_infections
            othr_symptomatic = rng.binomial(n=1, p=config.PROP_COVID_SYMPTOMATIC, size=n_othr).astype(bool) & othr_infections

            # work cases that are covid positive in either way
            work_tested_symptomatic = work_contacts_isolated & work_infections & work_symptomatic
            work_tested_asymptomatic = work_contacts_isolated & work_infections & ~work_symptomatic
            work_tested_positive = work_tested_symptomatic & work_tested_asymptomatic

            # other contacts that are positive in either way
            othr_tested_symptomatic = othr_contacts_isolated & othr_infections & othr_symptomatic
            othr_tested_asymptomatic = othr_contacts_isolated & othr_infections & ~othr_symptomatic
            othr_tested_positive = othr_tested_symptomatic & othr_tested_asymptomatic


        total_tests_performed = 0
        # count own test
        # TODO: Janky - if no contact tracing is going on, do NOT test the person
        if (do_app_tracing or do_manual_tracing or isolate_contacts_on_positive or isolate_contacts_on_symptoms):
            total_tests_performed += 1
        
        # If house isolated on symptoms, or on positive
        # These tests will not count against the primary case, as these would have been tested regardless.
        if case.covid:
            total_tests_performed += 0. # home_contacts_isolated.sum()

        # If contacts isolated on symptoms, or on positive
        # TODO: Again, after conversations, we will not test traced contacts unless a particular policy decision is made.
        # We do not count cases that would become positive and symptomatic against the primary case, but do count others. 
        if case.covid: # and test_contacts_on_positive:
            total_tests_performed += 0 # work_contacts_isolated.sum() + othr_contacts_isolated.sum()
        
        # Test contacts on positive test of the primary case. Only count the test excluding the symptomatic cases
        if test_contacts_on_positive and case.covid:
            total_tests_performed += (work_contacts_isolated & ~work_tested_symptomatic).sum()
            total_tests_performed += (othr_contacts_isolated & ~othr_tested_symptomatic).sum()

        ## Compute the quarantine days

        person_days_quarantine = 0

        # If person has covid, require full quarantine
        if case.covid and (isolate_individual_on_symptoms or isolate_individual_on_positive):
            person_days_quarantine += quarantine_length
        # If not, only require the test delay days of quarantine
        elif isolate_individual_on_symptoms:      
            person_days_quarantine += testing_delay
        ## Don't add any if: not isolating at all, individual only isolating after test complete

        # For household contacts, if isolating on symptoms and not covid, count test delay days 
        if isolate_household_on_symptoms and not case.covid:
            # TODO: only counts home contacts that actually isolate
            person_days_quarantine += testing_delay * home_contacts_isolated.sum()
        # If person has covid, whole house will have to do full lockdown for the period
        # TODO: might be able to let some out if the test negative?
        elif (isolate_household_on_positive or isolate_household_on_symptoms) and case.covid:
            person_days_quarantine += quarantine_length * home_contacts_isolated.sum()
        ## Don't add any if: Not isolating at all, or if waiting for positive test to isolate and doesn't have covid

        # For traced contacts, if isolating on positive and doesn't have covid, count test delay days
        ## NOTE: working with "quarantined" as this represents those traced who were still contacted and complied
        if isolate_contacts_on_symptoms and not case.covid:
            person_days_quarantine += testing_delay * (work_contacts_isolated.sum() + othr_contacts_isolated.sum())
        elif (isolate_contacts_on_positive or isolate_contacts_on_symptoms) and case.covid:
            # NOTE: for now assume that people are tested on the same day as isolated. So for contacts, same day as
            # the primary case if isolating on the day of symptoms, 3 days later if delaying. Will be the same number of days regardless.
            #  Probably would be an additional lag here.

            # If we are testing contacts on positive, then we will only need to quarantine those who are positive after the test
            # TODO: testing might be inefficient during latent period, perhaps we should quarantine contacts for latent_period and then test?
            if test_contacts_on_positive:
                # Those testing negative will spend 3 days in quarantine 
                person_days_quarantine += testing_delay * (work_contacts_isolated & ~work_tested_positive).sum()
                person_days_quarantine += testing_delay * (othr_contacts_isolated & ~othr_tested_positive).sum()
                # Those who test positive will go into full quarantine
                person_days_quarantine += quarantine_length * (work_contacts_isolated & work_tested_positive).sum()
                person_days_quarantine += quarantine_length * (othr_contacts_isolated & othr_tested_positive).sum()
            else:
                # Full quarantine for all contacts if not testing them
                person_days_quarantine += quarantine_length * (work_contacts_isolated.sum() + othr_contacts_isolated.sum())
                       
            
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


    ## Compute the base reproduction rate
    base_rr = home_infections.sum() + work_infections.sum() + othr_infections.sum()

    ## Compute the reproduction rate due to the policy
    # Remove infections due to case isolation
    home_infections_post_isolation = home_infections & ~home_contacts_prevented
    work_infections_post_isolation = work_infections & ~work_contacts_prevented
    othr_infections_post_isolation = othr_infections & ~othr_contacts_prevented

    # Count traced contacts as not included in the R TODO: make a proportion
    home_infections_post_policy = home_infections_post_isolation & ~home_contacts_isolated
    work_infections_post_policy = work_infections_post_isolation & ~work_contacts_isolated
    othr_infections_post_policy = othr_infections_post_isolation & ~othr_contacts_isolated
        
    ## Count fractional cases - will only occur if got tested
    if test_performed and fractional_infections:
        infectiousness_by_day = case.inf_profile
        cumulative_infectiousness = np.cumsum(infectiousness_by_day)
        infectious_period = len(infectiousness_by_day)

        # Get the days on which infections that were quarantined happened
        home_infection_days = home_infected_day[home_infections & home_contacts_isolated]
        work_infection_days = work_contacts[work_infections & work_contacts_isolated]
        othr_infection_days = othr_contacts[othr_infections & othr_contacts_isolated]

        # Get the people who were traced by the app
        work_contacts_trace_app_isolated = work_contacts_trace_app[work_infections & work_contacts_isolated]
        othr_contacts_trace_app_isolated = othr_contacts_trace_app[othr_infections & othr_contacts_isolated]

        # Trace delay is at max the manual trace delay
        work_trace_delay = manual_trace_delay * np.ones_like(work_infection_days)
        othr_trace_delay = manual_trace_delay * np.ones_like(othr_infection_days)

        # Contacts found via the app are traced with app_delay - assumed to be faster. (0)
        work_trace_delay[work_contacts_trace_app_isolated] = app_trace_delay
        othr_trace_delay[othr_contacts_trace_app_isolated] = app_trace_delay

        # Home contacts traced immediately
        home_trace_delay = np.zeros_like(home_infection_days)

        # Compute day of contact becoming infectious after case started being infectious
        home_infectious_start = home_infection_days + latent_period
        work_infectious_start = work_infection_days + latent_period
        othr_infectious_start = othr_infection_days + latent_period

        # Compute the days home cases are left out in the world infectious
        if isolate_household_on_symptoms:
            home_infections_days_not_quarantined = (test_perform_day + home_trace_delay) - home_infectious_start
        elif isolate_household_on_positive:
            home_infections_days_not_quarantined = (test_results_day + home_trace_delay) - home_infectious_start
        else:
            # If neither of these are true, then the case would not have made it to here as would have been in home_infections_post_policy
            home_infections_days_not_quarantined = (len(cumulative_infectiousness)) * np.ones(len(home_infectious_start), dtype=int)

        # Compute the days a work/othr case is left out in the world infectious
        if isolate_contacts_on_symptoms:
            work_infections_days_not_quarantined = (test_perform_day + work_trace_delay) - work_infectious_start
            othr_infections_days_not_quarantined = (test_perform_day + othr_trace_delay) - othr_infectious_start
        elif isolate_contacts_on_positive:
            work_infections_days_not_quarantined = (test_results_day + work_trace_delay) - work_infectious_start
            othr_infections_days_not_quarantined = (test_results_day + othr_trace_delay) - othr_infectious_start
        else:
            work_infections_days_not_quarantined = (len(cumulative_infectiousness)) * np.ones(len(work_infectious_start), dtype=int)
            othr_infections_days_not_quarantined = (len(cumulative_infectiousness)) * np.ones(len(othr_infectious_start), dtype=int)

        # clip the infectious period to max out at 1
        home_infections_days_not_quarantined[home_infections_days_not_quarantined > len(cumulative_infectiousness)] = len(cumulative_infectiousness)
        work_infections_days_not_quarantined[work_infections_days_not_quarantined > len(cumulative_infectiousness)] = len(cumulative_infectiousness)
        othr_infections_days_not_quarantined[othr_infections_days_not_quarantined > len(cumulative_infectiousness)] = len(cumulative_infectiousness)

        fractional_num_home = len(home_infections_days_not_quarantined)
        fractional_num_work = len(work_infections_days_not_quarantined)
        fractional_num_othr = len(othr_infections_days_not_quarantined)
        fractional_num = fractional_num_home + fractional_num_work + fractional_num_othr

        # Only care about ones where there is more than zero days spent unisolated
        home_infections_days_not_quarantined = home_infections_days_not_quarantined[home_infections_days_not_quarantined > 0]
        work_infections_days_not_quarantined = work_infections_days_not_quarantined[work_infections_days_not_quarantined > 0]
        othr_infections_days_not_quarantined = othr_infections_days_not_quarantined[othr_infections_days_not_quarantined > 0]

        # Add one to get indexing correct - 1st day infectious is 0 in array
        home_cumulative_infectiousness = cumulative_infectiousness[home_infections_days_not_quarantined - 1].sum()
        work_cumulative_infectiousness = cumulative_infectiousness[work_infections_days_not_quarantined - 1].sum()
        othr_cumulative_infectiousness = cumulative_infectiousness[othr_infections_days_not_quarantined - 1].sum()

        fractional_R = home_cumulative_infectiousness + work_cumulative_infectiousness + othr_cumulative_infectiousness
        inverse_fractional_R = fractional_num  - fractional_R
        home_fractional_R = home_cumulative_infectiousness
        home_inverse_fractional_R = fractional_num_home - home_cumulative_infectiousness
    else:
        fractional_R = 0.
        home_cumulative_infectiousness = 0.
        inverse_fractional_R = 0.
        home_fractional_R = 0.
        home_inverse_fractional_R = 0.
        

    # Count the reduced infection rate
    reduced_rr = home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum() + fractional_R

    social_distancing_infections_prevented = (work_contacts_wfh_limited & work_infections).sum() + (othr_contacts_limited & othr_infections).sum()
    symptom_isolation_infections_prevented = (home_contacts_prevented & home_infections).sum() + (work_contacts_prevented & work_infections).sum() + (othr_contacts_prevented & othr_infections).sum() + home_inverse_fractional_R - social_distancing_infections_prevented
    contact_tracing_infections_prevented   = base_rr - reduced_rr - social_distancing_infections_prevented - symptom_isolation_infections_prevented

    return {
            RETURN_KEYS.base_r: base_rr if case.covid else np.nan,
            RETURN_KEYS.reduced_r: reduced_rr if case.covid else np.nan,
            RETURN_KEYS.man_trace: manual_traces,
            RETURN_KEYS.app_trace: app_traces,
            RETURN_KEYS.tests: total_tests_performed,
            RETURN_KEYS.quarantine: person_days_quarantine,
            RETURN_KEYS.covid: case.covid,
            RETURN_KEYS.symptomatic: case.symptomatic,
            RETURN_KEYS.tested: test_performed and do_symptom_testing,
            RETURN_KEYS.secondary_infections: home_infections.sum() + work_infections.sum() + othr_infections.sum(),

            RETURN_KEYS.cases_prevented_social_distancing: social_distancing_infections_prevented,
            RETURN_KEYS.cases_prevented_symptom_isolating: symptom_isolation_infections_prevented,
            RETURN_KEYS.cases_prevented_contact_tracing: contact_tracing_infections_prevented,
            RETURN_KEYS.fractional_r: fractional_R - home_cumulative_infectiousness,

            # RETURN_KEYS.num_primary_symptomatic: 1 if case.covid and case.symptomatic else np.nan,
            # RETURN_KEYS.num_primary_asymptomatic: 1 if case.covid and (not case.symptomatic) else np.nan,
            # RETURN_KEYS.num_primary: 1 if case.covid else np.nan,
            # RETURN_KEYS.num_primary_symptomatic_missed: 1 if case.covid and case.symptomatic and (not test_performed) else np.nan,
            # RETURN_KEYS.num_primary_asymptomatic_missed: 1 if case.covid and (not case.symptomatic) and (not test_performed) else np.nan,
            # RETURN_KEYS.num_primary_missed: 1 if case.covid and (not test_performed) else np.nan,
            # RETURN_KEYS.num_secondary_from_symptomatic: home_infections_post_isolation.sum() + work_infections_post_isolation.sum() + othr_infections_post_isolation.sum() if case.covid and case.symptomatic else np.nan,
            # RETURN_KEYS.num_secondary_from_asymptomatic: home_infections_post_isolation.sum() + work_infections_post_isolation.sum() + othr_infections_post_isolation.sum() if case.covid and (not case.symptomatic) else np.nan,
            # RETURN_KEYS.num_secondary: home_infections_post_isolation.sum() + work_infections_post_isolation.sum() + othr_infections_post_isolation.sum() if case.covid else np.nan,
            # RETURN_KEYS.num_secondary_from_symptomatic_missed: home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum() if case.covid and case.symptomatic else np.nan,
            # RETURN_KEYS.num_secondary_from_asymptomatic_missed: home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum() if case.covid and (not case.symptomatic) else np.nan,
            # RETURN_KEYS.num_secondary_missed: home_infections_post_policy.sum() + work_infections_post_policy.sum() + othr_infections_post_policy.sum() if case.covid else np.nan,
        }
