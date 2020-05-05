from functools import partial

_contacts_configs = {
        "kucharski": dict(
            # infectivity
            home_sar=0.2,
            work_sar=0.06,
            other_sar=0.06,
            # For some reason this is 5 in Kucharski paper,
            # but there are 6 options for
            # noticing symptoms in p_day_noticed_symptoms.
            period=5  # Period of the simulation
            )


def get_contacts_config(name, _cfg_dct=_contacts_configs):
    try:
        return _cfg_dct[name.lower()]
    except KeyError:
        raise ValueError(
                f"Could not find config {name} in config.py."
                "Available configs are: {list(_cfg_dct.keys()}"
                )


_case_configs = {
        "kucharski": dict(
            p_under18=0.21,
            # following Kucharski.
            # This is currently independent from everything else.

            p_symptomatic_covid_neg=0, # 200 / 260
            p_symptomatic_covid_pos=0.6, # 30 / 260
            p_asymptomatic_covid_pos=0.4, # 30 / 260

            #Conditional on symptomatic
            p_has_app=0.35,
            # Conditional on having app
            p_report_app=0.75,
            p_report_nhs_g_app=0.5,

            # Conditional on not having app
            p_report_nhs_g_no_app=0.5,
            
            # Distribution of day on which the case notices their symptoms
            # This is conditinal on them being symptomatic at all
            p_day_noticed_symptoms=[0, 0.25, 0.25, 0.2, 0.3, 0]
        ),
        "anne": dict(
            p_under18=0.21,
            # following Kucharski.
            # This is currently independent from everything else.

            p_symptomatic_covid_neg=200 / 260,
            p_symptomatic_covid_pos=30 / 260,
            p_asymptomatic_covid_pos=30 / 260,

            #Conditional on symptomatic
            p_has_app=0.35,
            # Conditional on having app
            p_report_app=0.75,
            p_report_nhs_g_app=0.5,

            # Conditional on not having app
            p_report_nhs_g_no_app=0.5,
            
            # Distribution of day on which the case notices their symptoms
            # This is conditinal on them being symptomatic at all
            p_day_noticed_symptoms=[0, 0.25, 0.25, 0.2, 0.3, 0]
        ),

    }


get_case_config = partial(get_contacts_config, _cfg_dct=_case_configs)



_policy_config = {
        "cmmid":
        {
            "no_measures":
                {
                    "do_isolation": False,
                    "do_manual_tracing": False,
                    "do_app_tracing": False,
                    "do_pop_testing": False,
                },
            "isolation_only":
                {
                    "do_isolation": True,
                    "do_manual_tracing": False,
                    "do_app_tracing": False,
                    "do_pop_testing": False,
                    },
            "hh_quaratine_only":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": False,
                    "do_pop_testing": False,

                    "manual_work_trace_prob": 0.,
                    "manual_othr_trace_prob": 0.,
                },
             "hh_work_only":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": False,
                    "do_pop_testing": False,

                    "met_before_w": 1.,
                    "met_before_o": 1.,
                    "manual_othr_trace_prob": 0.,
                },
            "isolation_manual_tracing_met_limit":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": False,
                    "do_pop_testing": False,

                    "max_contacts": 4
                },
            "isolation_manual_tracing_met_only":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": False,
                    "do_pop_testing": False,
                },
            "isolation_manual_tracing":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": False,
                    "do_pop_testing": False,

                    "met_before_w": 1.,
                    "met_before_o": 1.
                },
            "cell_phone":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": True,
                    "do_pop_testing": False,

                    "met_before_w": 1.,
                    "met_before_o": 1.
                },
            "cell_phone_met_limit":
                {
                    "do_isolation": True,
                    "do_manual_tracing": True,
                    "do_app_tracing": True,
                    "do_pop_testing": False,

                    "met_before_w": 1.,
                    "met_before_o": 1.,
                    "max_contacts": 4
                },
            "pop_testing":
                {
                    "do_isolation": False,
                    "do_manual_tracing": False,
                    "do_app_tracing": False,
                    "do_pop_testing": True,

                    "p_pop_test": 0.05,
                },
        }
    }



_global_defaults = dict(
        do_isolation=True,    # Impose isolation on symptomatic persons
        do_manual_tracing=True,   # Perform manual contact tracing 
        do_app_tracing=True,  # Perform app-based contact tracing. ALT - could set phone prop to 0 if not active
        do_pop_testing=False, # Randomly test a proportion of the populatuion
        do_schools_open=True, # If schools in the country are open or not

        manual_home_trace_prob=1.,   # Probability of home contacts traces
        manual_work_trace_prob=0.95, # Probability of tracing a work contact
        manual_othr_trace_prob=0.95, # Probability of tracing an other contact

        met_before_w=0.79, # At work. At school=90%, which is defined in function later on
        met_before_s=0.9,  # At school. Will replace at work for under 18's
        met_before_h=1,    # Within HH
        met_before_o=0.52, # In other settings

        max_contacts=2e3,     # Enforced limit on number of contacts. Default of 200 to represent no limits
        wfh_prob=0,           # Probability people are working from home
        app_cov=0.53,         # App coverage
        p_pop_test=0.05,      # Proportion mass tested (5% per week)
        policy_adherence=0.9, # Adherence to testing/trace and quarantine
    )


_policy_config = {
        name: {k: dict(_global_defaults, **params) for k, params in strat.items()}
        for name, strat in _policy_config.items()
    }


def get_strategy_config(strat, cfg_names, _cfg_dct=_policy_config):
    try:
        strategy = _cfg_dct[strat.lower()]
    except KeyError:
        raise ValueError(f"Cannot find strategy {strat} in config.py")
    else:
        if cfg_names == "all":
            return dict(**strategy)
        else:
            output = dict()
            for cfg_name in cfg_names:
                try:
                    output[cfg_name] = strategy[cfg_name]
                except KeyError:
                    raise ValueError(f"Cannot find configuration {cfg_name} under "
                            "strategy {strat} in config.py")
            return output


