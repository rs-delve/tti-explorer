import numpy as np
from tqdm import tqdm

adult_survey_contacts = np.genfromtxt('data/contact_distributions_o18.csv',delimiter=',')[1:]
child_survey_contacts = np.genfromtxt('data/contact_distributions_u18.csv',delimiter=',')[1:]

def simulate_individual(
    key, # Random key
    adult_survey_contacts=adult_survey_contacts,
    child_survey_contacts=child_survey_contacts,
    # scenario_pick = 'no_measures'. Do in boilerplate
    # max_low_fix = 4,      # Social distancing limit in these scenarios. replace with max_contacts
    do_isolation = True,    # Impose isolation on symptomatic persons
    do_manual_tracing = True,      # Perform manual contact tracing 
    do_app_tracing = True,  # Perform app-based contact tracing. ALT - could set phone prop to 0 if not active
    do_pop_testing = False,  # Randomly test a proportion of the populatuion
    do_schools_open = True,    # If schools in the country are open or not

    home_trace_prob = 1.,   # Probability of home contacts traces
    work_trace_prob = 0.95, # Probability of tracing a work contact
    othr_trace_prob = 0.95, # Probability of tracing an other contact

    met_before_w = 0.79, # At work. At school = 90%, which is defined in function later on
    met_before_s = 0.9,  # At school. Will replace at work for under 18's
    met_before_h = 1,    # Within HH
    met_before_o = 0.52, # In other settings

    home_risk = 0.2,        # Risk of infection to household members
    non_home_risk = 0.06,   # Risk of infection to non-household members

    symp_prob = 0.6,        # Probability of symptomatic
    asymp_trans_prob = 0.5, # Probability of asymptomatic transmission 

    max_contacts = 2e3,     # Enforced limit on number of contacts. Default of 200 to represent no limits
    wfh_prob = 0,           # Probability people are working from home
    app_cov = 0.53,         # App coverage
    isolate_distn = [0,0.25,0.25,0.2,0.3,0], # distribution of time to isolate (1st day presymp)

    pt_extra = 0,           # Optional extra transmission intervention
    pt_extra_reduce = 0,    # Reduction from extra intervention
):
    np.random.seed(key)
   
    ##### Assumed Constants #####

    under_18_prob = 0.21
    trace_adherence = 0.9 # Adherence to testing/quarantine

    phone_coverage = app_cov # App coverage in non-app scenarios - no traces from app.
    p_pop_test = 0.05 # Proportion mass tested (5% per week)
    inf_period = 5 # Infectious period
    
    # Proportion of contacts met before

    
    # Set contact limit default high to avoid censoring in default scenarios
    # max_contacts = 2e3 

    p_tested = trace_adherence # Proportion who get tested when they have symptoms
    time_isolate = isolate_distn # Distribution over symptomatic period
    p_symptomatic = symp_prob
    transmission_asymp = asymp_trans_prob

    ### TEST DATA

    data_user_col_red_o18 = adult_survey_contacts
    data_user_col_red_u18 = child_survey_contacts

    ###

    #############################
    
    sample = np.random.uniform()
    if sample < under_18_prob:
        u_18 = True
        n_user_age = len(data_user_col_red_u18)
        data_ii = data_user_col_red_u18[np.random.randint(n_user_age)]  # Sample row from contact survey
        met_before_w = met_before_s
        wfh_true = not do_schools_open    # Children wfh if schools shut
    else:
        u_18 = False
        n_user_age = len(data_user_col_red_o18)
        data_ii = data_user_col_red_o18[np.random.randint(n_user_age)]  # Sample row from contact survey
        met_before_w = met_before_w
        wfh_true = np.random.uniform() < wfh_prob  # Sample percentage of adults working from home


    phone_T = np.random.uniform() < phone_coverage  # Check if has phone app
    symp_T = np.random.uniform() < symp_prob        # Check if symptomatic
    tested_T = np.random.uniform() < p_tested       # Check if gets tested if had symptoms
    # inf_scale_mass_pop = 1.
    extra_red = 1.

    # Reduce infectivity if asymptomatic
    if symp_T:
        inf_propn = 1.0
    else:
        inf_propn = transmission_asymp

    inf_period_ii = np.random.choice(len(time_isolate), p=time_isolate)     # Will sample 0 TODO: is this correct?
    if not (tested_T and symp_T and do_isolation): 
        inf_period_ii = inf_period   # If not the correct combination of events, will not isolate

    # Check if caught by random testing of population
    pop_test_inf_period = np.random.choice(7)
    if not (do_pop_testing and (np.random.uniform() < p_pop_test)):
        pop_test_inf_period = inf_period

    inf_period_ii = np.minimum(inf_period_ii, pop_test_inf_period)  # Check which of random testing or normal tracing will result in isolation first

    # TODO This precludes manual tracing if they have the app
    if do_app_tracing:
        if phone_T:
            work_trace_prob = phone_coverage
            othr_trace_prob = phone_coverage
        else:
            work_trace_prob = 0.
            othr_trace_prob = 0.

    if (np.random.uniform() < pt_extra):
        tested_T = True
        extra_red = 1-pt_extra_reduce

    inf_ratio = inf_period_ii / inf_period

    ## Start computing infections caused and tracing of individuals.

    # Compute new contacts, assume home are all the same, work and other all unique.
    home_contacts = data_ii[0]
    work_contacts = data_ii[1] * inf_period # TODO: should we compute UNIQUE contacts, based on the met before values?
    othr_contacts = data_ii[2] * inf_period

    scale_other = min(1, (max_contacts * inf_period) / othr_contacts)

    # Sample the number of people that would be infected with no policy
    home_infect_basic = np.random.binomial(p=home_risk * inf_propn, n=home_contacts)
    work_infect_basic = np.random.binomial(p=(non_home_risk * inf_propn), n=work_contacts)
    othr_infect_basic = np.random.binomial(p=(non_home_risk * inf_propn), n=othr_contacts)
    rr_basic = home_infect_basic + work_infect_basic + othr_infect_basic

    if wfh_true:
        inf_ratio_work = 0
    else:
        inf_ratio_work = inf_ratio

    # Sample the reduction in infections due to national policy (not including contact tracing)
    home_infect_policy = np.random.binomial(p=(inf_ratio), n=home_infect_basic)
    work_infect_policy = np.random.binomial(p=(inf_ratio_work), n=work_infect_basic)
    othr_infect_policy = np.random.binomial(p=(inf_ratio * scale_other * extra_red), n=othr_infect_basic)
    rr_national_policy = home_infect_policy + work_infect_policy + othr_infect_policy

    # Sample the number of contact successfully traced who were NOT infected 
    home_uninfected_traced = np.random.binomial(p=home_trace_prob,n= home_contacts-home_infect_policy)
    work_uninfected_traced = np.random.binomial(p=work_trace_prob * met_before_w, n=work_contacts-work_infect_policy)
    othr_uninfected_traced = np.random.binomial(p=othr_trace_prob * met_before_o, n=othr_contacts-othr_infect_policy)
    
    # Sample the number of contacts successfully traced who WERE infected
    home_infected_traced = np.random.binomial(p=home_trace_prob, n=home_infect_policy)
    work_infected_traced = np.random.binomial(p=work_trace_prob * met_before_w, n=work_infect_policy)
    othr_infected_traced = np.random.binomial(p=othr_trace_prob * met_before_o, n=othr_infect_policy)

    # Account for non-compliance to trace isolation
    home_infections_averted = np.random.binomial(p=trace_adherence, n=home_infected_traced)
    work_infections_averted = np.random.binomial(p=trace_adherence, n=work_infected_traced)
    othr_infections_averted = np.random.binomial(p=trace_adherence, n=othr_infected_traced)

    if tested_T and symp_T and (do_manual_tracing | do_app_tracing):
        total_averted = home_infections_averted + work_infections_averted + othr_infections_averted
    else:
        total_averted = 0

    rr_contact_trace = rr_national_policy - total_averted

    total_traced = home_uninfected_traced + home_infected_traced + work_uninfected_traced + work_infected_traced + othr_uninfected_traced + othr_infected_traced

    if home_contacts == 0: home_contacts += 1
    if work_contacts == 0: work_contacts += 1
    if othr_contacts == 0: othr_contacts += 1

    # return np.array([home_infect_basic / home_contacts, work_infect_basic / work_contacts, othr_infect_basic / othr_contacts, rr_basic, rr_national_policy, rr_contact_trace])
    return np.array([rr_basic, rr_contact_trace, total_averted])


scenarios = [
    'no_measures',
    'isolation_only',
    'hh_quaratine_only',
    'hh_work_only',
    'isolation_manual_tracing_met_limit',
    'isolation_manual_tracing_met_only',
    'isolation_manual_tracing',
    'cell_phone',
    'cell_phone_met_limit',
    'pop_testing',
    'pt_extra',
]

def build_scenario_args(scenario):

    additional_args = {}

    if scenario == 'no_measures':
        additional_args['do_isolation'] = False
        additional_args['do_manual_tracing'] = False
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

    elif scenario == 'isolation_only':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = False
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

    elif scenario == 'hh_quaratine_only':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

        additional_args['work_trace_prob'] = 0.
        additional_args['othr_trace_prob'] = 0.

    elif scenario == 'hh_work_only':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

        additional_args['met_before_w'] = 1.
        additional_args['met_before_o'] = 1.
        additional_args['othr_trace_prob'] = 0.

    elif scenario == 'isolation_manual_tracing_met_limit':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

        additional_args['max_contacts'] = 4.

    elif scenario == 'isolation_manual_tracing_met_only':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

    elif scenario == 'isolation_manual_tracing':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

        additional_args['met_before_w'] = 1.
        additional_args['met_before_o'] = 1.
    
    elif scenario == 'cell_phone':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = True
        additional_args['do_pop_testing'] = False
        
        additional_args['met_before_w'] = 1.
        additional_args['met_before_o'] = 1.

    elif scenario == 'cell_phone_met_limit':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = True
        additional_args['do_pop_testing'] = False

        additional_args['met_before_w'] = 1.
        additional_args['met_before_o'] = 1.
        additional_args['max_contacts'] = 4.

    elif scenario == 'pop_testing':
        additional_args['do_isolation'] = False
        additional_args['do_manual_tracing'] = False
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = True

    elif scenario == 'pt_extra':
        additional_args['do_isolation'] = True
        additional_args['do_manual_tracing'] = True
        additional_args['do_app_tracing'] = False
        additional_args['do_pop_testing'] = False

        additional_args['work_trace_prob'] = 0.
        additional_args['othr_trace_prob'] = 0.

        additional_args['pt_extra'] = 0.
        additional_args['pt_extra_reduce'] = 0.

    else:
        raise ValueError(f'scenario {scenario} does not match a known scenario')

    return additional_args


def run_scenarios(scenario_list, n_runs=50000, **kwargs):
    for scenario in tqdm(scenario_list, desc='Running scenarios', position=0):
        results = np.array([
            simulate_individual(i, **build_scenario_args(scenario), **kwargs) for i in tqdm(range(50000), desc='Simulating individuals', position=1)
        ])
        # print(scenario, np.mean(results, axis=0))



kucharski_scenarios = [
    "no_measures","isolation_only","hh_quaratine_only","hh_work_only",
                     "isolation_manual_tracing_met_only","isolation_manual_tracing_met_limit",
                     "isolation_manual_tracing","cell_phone","cell_phone_met_limit",
                     "pop_testing","pt_extra"
]


# results = np.array([
#     simulate_individual(i, **build_scenario_args('isolation_manual_tracing_met_limit')) for i in range(execs)
# ])

# print(np.mean(results, axis=0))

# run_scenarios(kucharski_scenarios)

for scenario in kucharski_scenarios:
    results = np.array([
        simulate_individual(i, **build_scenario_args(scenario)) for i in range(50000)
    ])
    print(scenario, np.nanmean(results, axis=0))


