import jax.numpy as np
from jax import random, vmap, lax
# imports that Bobby used for testing
# import pandas as pd
# from jax.interpreters.xla import DeviceArray
# prob_symp = 0.9
# prob_t_asymp = 0.9
# do_pop_test = True
# data_user_col_red_o18 = np.array(pd.read_csv('/home/bobby/delve/tti-explorer/data/contact_distributions_o18.csv', sep=','))
# data_user_col_red_u18 = np.array(pd.read_csv('/home/bobby/delve/tti-explorer/data/contact_distributions_u18.csv', sep=','))
# cell_phone_bool = True
# pt_extra_bool = True
# do_isolation = True
####

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
    met_before_w = 0.79 # At work. At school = 90%, which is defined in function later on. We don't need this
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

    def sample_user(inputs):
        met_before_w, wfh_t, age_data = inputs
        n_user_age = len(age_data)
        pick_user = random.randint(key, shape = (1,), minval = 0, maxval = n_user_age)
        data_ii = lax.index_take(age_data, (pick_user,), axes=(0,)) if isinstance(age_data, DeviceArray) else np.take(age_data, pick_user, axis=0)
        return data_ii, met_before_w, wfh_t

    # Sample user
    key, unif_rv = uniform_sample(key)
    o18_wfh = lax.cond(unif_rv < wfh_prob, None, lambda x: True, None, lambda x: False)
    u18_inputs = (0.9, False, data_user_col_red_u18)
    o18_inputs = (0.79, o18_wfh, data_user_col_red_o18)
    key, unif_rv = uniform_sample(key)
    data_ii, met_before_w, wfh_t = lax.cond(unif_rv < under_18_prob, u18_inputs, sample_user, o18_inputs, sample_user)

    # Simulate infectious period
    # Decide if symptomatic and tested`
    key, unif_rv = uniform_sample(key)
    phone_T = lax.cond(unif_rv < phone_coverage, None, lambda x: True, None, lambda x: False) # has phone app?
    key, unif_rv = uniform_sample(key)
    symp_T = lax.cond(unif_rv < p_symptomatic, None, lambda x: True, None, lambda x: False) # symptomatic?
    key, unif_rv = uniform_sample(key)
    tested_T = lax.cond(unif_rv < p_tested, None, lambda x: True, None, lambda x: False) # would be tested
    inf_scale_mass_pop = 1
    extra_red = 1

    # Set infectious period based on symptomatic & tested
    time_isolate_logits = np.log(np.array(time_isolate))
    inf_period_ii = random.categorical(key, time_isolate_logits)
    key, subkey = random.split(key)
    inf_period_ii = lax.cond(np.logical_and((do_isolation and symp_T),  tested_T), None, lambda x: inf_period_ii, None, lambda x: inf_period)

    # Set infectious period for population test_single_nll
    pop_inf_period_probs = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 2/7])
    pop_inf_period_logits = np.log(pop_inf_period_probs)
    pop_inf_period = random.categorical(key, pop_inf_period_logits)
    key, unif_rv = uniform_sample(key)
    pop_tested_in = (True, pop_inf_period)
    not_pop_tested_in = (tested_T, inf_period)
    tested_T, pop_inf_period_ii = lax.cond(do_pop_test and unif_rv < p_pop_test, None, lambda x: pop_tested_in, None, lambda x: not_pop_tested_in) # Need to define do_pop_test
    inf_period_ii = np.minimum(inf_period_ii, pop_inf_period_ii)

    # Set relative transmission of asymptomatics
    inf_propn = lax.cond(symp_T, None, lambda x: 1., None, lambda x: transmission_asymp)

    # Check if contacts phone traced (in cell phone scenario), need to define cell_phone_bool to be True if scenario == 'cell_phone' or 'cell_ohne_met_limit'
    ww_trace, other_trace = lax.cond(cell_phone_bool and phone_T, (phone_coverage, phone_coverage), lambda x: x, (0., 0.), lambda x: x)

    # Extra transmission reduction
    key, unif_rv = uniform_sample(key)
    tested_T, extra_red = lax.cond(pt_extra_bool and unif_rv < pt_extra, (True, 1-pt_extra_reduce), lambda x: x, (tested_T, extra_red), lambda x: x)

    # Proportion infectious
    inf_ratio = inf_period_ii / inf_period

    ## Commutation strategy
