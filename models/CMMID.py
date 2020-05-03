import jax.numpy as np
from jax import random, vmap, lax


def simulate_individual(
    key, # Random key
    # scenario_pick = 'no_measures'. Do in boilerplate
    # max_low_fix = 4,      # Social distancing limit in these scenarios. replace with max_contacts
    max_contacts = 2e3,     # Enforced limit on number of contacts. Default of 200 to represent no limits
    wfh_prob = 0,           # Probability people are working from home
    # trace_prop = 0.95,    # Proportion of contacts traced. Replace with banded variables
    home_trace_prob = 1.,   # Probability of home contacts traces
    work_trace_prob = 0.95, # Probability of tracing a work contact
    othr_trace_prob = 0.95, # Probability of tracing an other contact
    app_cov = 0.53,         # App coverage
    symp_prob = 0.6,        # Probability of symptomatic
    asymp_trans_prob = 0.5, # Probability of asymptomatic transmission 
    isolate_distn = [0,0.25,0.25,0.2,0.3,0], # distribution of time to isolate (1st day presymp)
    pt_extra = 0,           # Optional extra transmission intervention
    pt_extra_reduce = 0,    # Reduction from extra intervention
    home_risk = 0.2,        # Risk of infection to household members
    non_home_risk = 0.06,   # Risk of infection to non-household members
    do_isolation = True,    # Impose isolation on symptomatic persons
    do_manual_tracing = True,      # Perform manual contact tracing 
    do_app_tracing = True,  # Perform app-based contact tracing.

    # range_n = None,       # Pick specific scenarios to run. Do in boilerplate
    # dir_pick = "",        # Output directory, do in boilerplate
    # output_r = False,     # Do in boilerplate
    n_run = 5e3,          # Number of simulations, do in boilerplate
)
   
    ##### Assumed Constants #####

    under_18_prob = 0.21
    trace_adherence = 0.9 # Adherence to testing/quarantine

    phone_coverage = 0. # App coverage in non-app scenarios - no traces from app.
    p_pop_test = 0.05 # Proportion mass tested (5% per week)
    inf_period = 5 # Infectious period
    
    # Proportion of contacts met before
    met_before_w = 0.79 # At work. At school = 90%, which is defined in function later on
    met_before_s = 0.9  # At school. Will replace at work for under 18's
    met_before_h = 1    # Within HH
    met_before_o = 0.52 # In other settings
    
    # Set contact limit default high to avoid censoring in default scenarios
    max_contacts = 2e3 

    #############################

    p_tested = trace_adherence # Proportion who get tested
    time_isolate = isolate_distn # Distribution over symptomatic period
    p_symptomatic = prob_symp
    transmission_asymp = prob_t_asymp
    
    
    # Tracing parameters
    # hh_trace = home_trace_prob # Tracing in house hold
    # ww_trace = work_trace_prob # Tracing at work
    # other_trace = othr_trace_prob # Tracing others

    ## Commutation strategy