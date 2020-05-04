import jax
import jax.numpy as np
from jax import random, vmap, lax, jit
from jax.interpreters.xla import DeviceArray

@jit
def factorial(x):
    return np.exp(jax.scipy.special.gammaln(x+1))

@jit
def nCk(n, k):
    return factorial(n) / (factorial(n-k) * factorial(k))

@jit
def uniform_sample(key):
    key, split_key = random.split(key)
    return key, random.uniform(split_key)


def binomial_sample(key, p, n): # 1-D only
    k = np.arange(100)

    pdf = nCk(n, k) * (p ** k) * ((1-p) ** (n-k))

    cdf = np.cumsum(pdf, axis=0)

    unif = random.uniform(key)

    cdf = np.greater(cdf, unif)

    index = np.argmax(cdf)

    return index

    # samples = random.bernoulli(key, p=np.atleast_1d(p), shape=[100])
    # if isinstance(samples, DeviceArray):
    #     selected = lax.dynamic_slice_in_dim(samples, np.array([0]), np.array([n]))
    # else:
    #     idx = np.arange(100)
    #     idxs = lax.dynamic_slice_in_dim(idx, 0, n)
    #     selected = np.take(samples, idxs, axis=0)
    # return selected.sum()
    # return np.atleast_1d(random.bernoulli(key, p=np.atleast_1d(p), shape=[n]).astype(int).sum())
    #   
# binomial_sample = jit(binomial_sample)

@jit
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
):
   
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
    p_symptomatic = symp_prob
    transmission_asymp = asymp_trans_prob
    
    
    # Tracing parameters
    # hh_trace = home_trace_prob # Tracing in house hold
    # ww_trace = work_trace_prob # Tracing at work
    # other_trace = othr_trace_prob # Tracing others

    ## Commutation strategy


    ##### TEST NUMBERS DEFINITIONS

    data_ii = np.array([300,600,200])
    inf_propn = 1.0
    wfh_true = True
    inf_ratio = 0.6
    extra_red = 1.0
    tested_T = True
    symp_T = True
    do_tracing = True
    #####


    ## Start computing infections caused and tracing of individuals.

    home_contacts = data_ii[0]
    work_contacts = data_ii[1] * inf_period # TODO: should we compute UNIQUE contacts, based on the met before values?
    othr_contacts = data_ii[2] * inf_period

    # home_contacts = 4
    # work_contacts = 4
    # othr_contacts = 4

    scale_other = min(1, (max_contacts * inf_period) / othr_contacts)

    # Sample the number of people that would be infected with no policy
    key, home_key, work_key, othr_key = random.split(key, 4)
    home_infect_basic = binomial_sample(home_key, home_risk * inf_propn, home_contacts)
    work_infect_basic = binomial_sample(work_key, (non_home_risk * inf_propn), work_contacts)
    othr_infect_basic = binomial_sample(othr_key, (non_home_risk * inf_propn), othr_contacts)
    rr_basic = home_infect_basic + work_infect_basic + othr_infect_basic

    key, sample = uniform_sample(key)
    inf_ratio_work = lax.cond(wfh_true, None, lambda x: 0., None, lambda x: inf_ratio)

    # Sample the reduction in infections due to national policy (not including contact tracing)
    key, home_key, work_key, othr_key = random.split(key, 4)
    home_infect_policy = binomial_sample(home_key, (inf_ratio), home_infect_basic)
    work_infect_policy = binomial_sample(work_key, (inf_ratio_work), work_infect_basic)
    othr_infect_policy = binomial_sample(othr_key, (inf_ratio * scale_other * extra_red), othr_infect_basic)
    rr_national_policy = home_infect_policy + work_infect_policy + othr_infect_policy

    # Sample the number of contact successfully traced who were NOT infected 
    key, home_key, work_key, othr_key = random.split(key, 4)
    home_uninfected_traced = binomial_sample(home_key, home_trace_prob, home_contacts-home_infect_policy)
    work_uninfected_traced = binomial_sample(work_key, work_trace_prob * met_before_w, work_contacts-work_infect_policy)
    othr_uninfected_traced = binomial_sample(othr_key, othr_trace_prob * met_before_o, othr_contacts-othr_infect_policy)
    
    # Sample the number of contacts successfully traced who WERE infected
    key, home_key, work_key, othr_key = random.split(key, 4)
    home_infected_traced = binomial_sample(home_key, home_trace_prob, home_infect_policy)
    work_infected_traced = binomial_sample(work_key, work_trace_prob * met_before_w, work_infect_policy)
    othr_infected_traced = binomial_sample(othr_key, othr_trace_prob * met_before_o, othr_infect_policy)

    # Account for non-compliance to trace isolation
    key, home_key, work_key, othr_key = random.split(key, 4)
    home_infections_averted = binomial_sample(home_key, trace_adherence, home_infect_policy)
    work_infections_averted = binomial_sample(work_key, trace_adherence, work_infect_policy)
    othr_infections_averted = binomial_sample(othr_key, trace_adherence, othr_infect_policy)

    total_averted = lax.cond(
        tested_T and symp_T and do_tracing,
        [home_infections_averted, work_infections_averted, othr_infections_averted],
        lambda x: np.sum(x),
        None,
        lambda x: 0
    )

    rr_contact_trace = rr_national_policy - total_averted

    total_traced = home_uninfected_traced + home_infected_traced + work_uninfected_traced + work_infected_traced + othr_uninfected_traced + othr_infected_traced

    return np.array([rr_basic, rr_national_policy, rr_contact_trace])



default_args = {
    'max_contacts' : 2e3,     # Enforced limit on number of contacts. Default of 200 to represent no limits
    'wfh_prob' : 0,           # Probability people are working from home
    'home_trace_prob' : 1.,   # Probability of home contacts traces
    'work_trace_prob' : 0.95, # Probability of tracing a work contact
    'othr_trace_prob' : 0.95, # Probability of tracing an other contact
    'app_cov' : 0.53,         # App coverage
    'symp_prob' : 0.6,        # Probability of symptomatic
    'asymp_trans_prob' : 0.5, # Probability of asymptomatic transmission 
    'isolate_distn' : [0,0.25,0.25,0.2,0.3,0], # distribution of time to isolate (1st day presymp)
    'pt_extra' : 0,           # Optional extra transmission intervention
    'pt_extra_reduce' : 0,    # Reduction from extra intervention
    'home_risk' : 0.2,        # Risk of infection to household members
    'non_home_risk' : 0.06,   # Risk of infection to non-household members
    'do_isolation' : True,    # Impose isolation on symptomatic persons
    'do_manual_tracing' : True,      # Perform manual contact tracing 
    'do_app_tracing' : True,  # Perform app-based contact tracing.
    'n_run' : 5e3,          # Number of simulations, do in boilerplate
}

print(simulate_individual(jax.random.PRNGKey(0), **default_args))

for i in range(20):
    key=jax.random.PRNGKey(i)
    # print(binomial_sample(key, p=0.5, n=50))

import timeit

pop_size = 20000

print(timeit.timeit(lambda : simulate_individual(jax.random.PRNGKey(0), **default_args), number=pop_size))

vmapped_fun = vmap(lambda x : simulate_individual(x, **default_args), 0)

main_key = random.PRNGKey(988)
pop_keys = random.split(main_key, pop_size)

print(timeit.timeit(lambda : vmapped_fun(pop_keys), number=1))