"""
Notes:
    - the nppl entry in infection_proportions is measured in thousands

"""
from collections import namedtuple
from functools import partial

import numpy as np

from .contacts import he_infection_profile

PROP_COVID_SYMPTOMATIC = 0.6

# used in run sensitivity
STATISTIC_COLNAME = "statistic"


_contacts_configs = {
    "kucharski": dict(
        # infectivity
        home_sar=0.2,
        work_sar=0.06,
        other_sar=0.06,
        # For some reason this is 5 in Kucharski paper,
        # but there are 6 options for
        period=5,
        # noticing symptoms in p_day_noticed_symptoms.
        asymp_factor=0.5,
    ),
    "delve": dict(
        home_sar=0.3, work_sar=0.045, other_sar=0.045, period=10, asymp_factor=0.5
    ),
}

_contacts_configs["delve-symptomatic"] = dict(**_contacts_configs["delve"])


def get_contacts_config(name, _cfg_dct=_contacts_configs):
    try:
        return _cfg_dct[name.lower()]
    except KeyError:
        raise ValueError(
            f"Could not find config {name} in config.py. "
            f"Available configs are: {list(_cfg_dct.keys())}"
        )


_case_configs = {
    # These are supposed to be exactly as with Kucharski paper
    "kucharski": dict(
        p_under18=0.21,
        # following Kucharski.
        # This is currently independent from everything else.
        infection_proportions={
            "dist": [
                0,
                PROP_COVID_SYMPTOMATIC,
                1 - PROP_COVID_SYMPTOMATIC,
            ],  # symp covid neg, symp covid pos, asymp covid pos
            "nppl": 1,  # shouldn't matter if everyone has covid
        },
        # Distribution of day on which the case notices their symptoms
        # This is conditinal on them being symptomatic at all
        p_day_noticed_symptoms=[0, 0.25, 0.25, 0.2, 0.3, 0],
        # length of this determines simulation length
        inf_profile=np.full(5, 1 / 5).tolist(),
    ),
    "delve": dict(
        p_under18=0.21,
        # following Kucharski.
        # This is currently independent from everything else.
        # symp covid neg, symp covid pos, asymp covid pos
        infection_proportions={
            "dist": [
                100 / 120,
                PROP_COVID_SYMPTOMATIC * 20 / 120,
                (1 - PROP_COVID_SYMPTOMATIC) * 20 / 120,
            ],
            "nppl": 120,
        },
        # Distribution of day on which the case notices their symptoms
        # This is conditinal on them being symptomatic at all
        p_day_noticed_symptoms=[
            0,
            0.25,
            0.25,
            0.2,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.00,
        ],  # mean delay 3.05 days
        # daily infectivity profile
        # length of this determines simulation length
        # should sum to 1
        inf_profile=(
            he_infection_profile(period=10, gamma_params={"a": 2.80, "scale": 1 / 0.69})
        ).tolist(),
    ),
    "delve-symptomatic": dict(
        p_under18=0.21,
        # following Kucharski.
        # This is currently independent from everything else.
        # symp covid neg, symp covid pos, asymp covid pos
        infection_proportions={
            "dist": [100 / 120, 1 * 20 / 120, 0 * 20 / 120],
            "nppl": 120,
        },
        # Distribution of day on which the case notices their symptoms
        # This is conditinal on them being symptomatic at all
        p_day_noticed_symptoms=[
            0,
            0.25,
            0.25,
            0.2,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.00,
        ],  # mean delay 3.05 days
        # daily infectivity profile
        # length of this determines simulation length
        # should sum to 1
        inf_profile=(
            he_infection_profile(period=10, gamma_params={"a": 2.80, "scale": 1 / 0.69})
        ).tolist(),
    ),
}


# for generating separate populations
for i, name in enumerate(
    ["delve-symp-covneg", "delve-symp-covpos", "delve-asymp-covpos"]
):
    _contacts_configs[name] = dict(**_contacts_configs["delve"])
    _case_configs[name] = dict(**_case_configs["delve"])
    _case_configs[name]["infection_proportions"] = [int(k == i) for k in range(3)]


get_case_config = partial(get_contacts_config, _cfg_dct=_case_configs)


S_levels = {
    "S5": {
        "isolate_individual_on_symptoms": True,
        "isolate_individual_on_positive": True,
        "isolate_household_on_symptoms": True,
        "isolate_household_on_positive": True,
        "do_symptom_testing": True,
        "met_before_w": 0.79,
        "met_before_s": 0.9,
        "met_before_o": 1,
        "wfh_prob": 0.65,
        "max_contacts": 1,
        "go_to_school_prob": 0.0,
    },
    "S4": {
        "isolate_individual_on_symptoms": True,
        "isolate_individual_on_positive": True,
        "isolate_household_on_symptoms": True,
        "isolate_household_on_positive": True,
        "do_symptom_testing": True,
        "met_before_w": 0.79,
        "met_before_s": 0.9,
        "met_before_o": 1,
        "wfh_prob": 0.55,
        "max_contacts": 4,
        "go_to_school_prob": 0.0,
    },
    "S3": {
        "isolate_individual_on_symptoms": True,
        "isolate_individual_on_positive": True,
        "isolate_household_on_symptoms": True,
        "isolate_household_on_positive": True,
        "do_symptom_testing": True,
        "met_before_w": 0.79,
        "met_before_s": 0.9,
        "met_before_o": 0.9,
        "wfh_prob": 0.45,
        "max_contacts": 10,
        "go_to_school_prob": 0.5,
    },
    "S2": {
        "isolate_individual_on_symptoms": True,
        "isolate_individual_on_positive": True,
        "isolate_household_on_symptoms": True,
        "isolate_household_on_positive": True,
        "do_symptom_testing": True,
        "met_before_w": 0.79,
        "met_before_s": 0.9,
        "met_before_o": 0.75,
        "wfh_prob": 0.25,
        "max_contacts": 20,
    },
    "S1": {
        "isolate_individual_on_symptoms": True,
        "isolate_individual_on_positive": True,
        "isolate_household_on_symptoms": True,
        "isolate_household_on_positive": True,
        "do_symptom_testing": True,
        "met_before_w": 0.79,
        "met_before_s": 0.9,
        "met_before_o": 0.52,
    },
    "S0": {
        "isolate_individual_on_symptoms": False,
        "isolate_individual_on_positive": False,
        "isolate_household_on_symptoms": False,
        "isolate_household_on_positive": False,
        "isolate_contacts_on_symptoms": False,
        "isolate_contacts_on_positive": False,
        "do_symptom_testing": False,
        "do_manual_tracing": False,
        "do_app_tracing": False,
    },
}

contact_trace_options = {
    "no_TTI": {
        "isolate_contacts_on_symptoms": False,
        "isolate_contacts_on_positive": False,
        "do_manual_tracing": False,
        "do_app_tracing": False,
    },
    "symptom_based_TTI": {
        "isolate_contacts_on_symptoms": True,
        "isolate_contacts_on_positive": True,
        "do_manual_tracing": True,
        "do_app_tracing": True,
    },
    "test_based_TTI": {
        "isolate_contacts_on_symptoms": False,
        "isolate_contacts_on_positive": True,
        "do_manual_tracing": True,
        "do_app_tracing": True,
    },
    "test_based_TTI_test_contacts": {
        "isolate_contacts_on_symptoms": False,
        "isolate_contacts_on_positive": True,
        "test_contacts_on_positive": True,
        "do_manual_tracing": True,
        "do_app_tracing": True,
    },
}


_policy_configs = {
    "cmmid": {
        "no_measures": {
            "do_isolation": False,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": False,
            # "manual_home_trace_prob": 0.,
            # "manual_work_trace_prob": 0.,
            # "manual_othr_trace_prob": 0.,
        },
        "isolation_only": {
            "do_isolation": True,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": False,
            # "manual_home_trace_prob": 0.,
            # "manual_work_trace_prob": 0.,
            # "manual_othr_trace_prob": 0.,
        },
        "hh_quaratine_only": {
            "do_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "manual_work_trace_prob": 0.0,
            "manual_othr_trace_prob": 0.0,
        },
        "hh_work_only": {
            "do_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "met_before_w": 1.0,
            "met_before_o": 1.0,
            "manual_othr_trace_prob": 0.0,
        },
        "isolation_manual_tracing_met_limit": {
            "do_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "max_contacts": 4,
        },
        "isolation_manual_tracing_met_only": {
            "do_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
        },
        "isolation_manual_tracing": {
            "do_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "met_before_w": 1.0,
            "met_before_o": 1.0,
        },
        "cell_phone": {
            "do_isolation": True,
            "do_manual_tracing": False,
            "do_app_tracing": True,
            "do_pop_testing": False,
            "met_before_w": 1.0,
            "met_before_o": 1.0,
        },
        "cell_phone_met_limit": {
            "do_isolation": True,
            "do_manual_tracing": False,
            "do_app_tracing": True,
            "do_pop_testing": False,
            "met_before_w": 1.0,
            "met_before_o": 1.0,
            "max_contacts": 4,
        },
        "pop_testing": {
            "do_isolation": False,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": True,
            "p_pop_test": 0.05,
        },
    },
    "cmmid_better": {
        "no_measures": {
            "do_individual_isolation": False,
            "do_household_isolation": False,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": False,
            # "manual_home_trace_prob": 0.,
            # "manual_work_trace_prob": 0.,
            # "manual_othr_trace_prob": 0.,
        },
        "isolation_only": {
            "do_individual_isolation": True,
            "do_household_isolation": False,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": False,
            # "manual_home_trace_prob": 0.,
            # "manual_work_trace_prob": 0.,
            # "manual_othr_trace_prob": 0.,
        },
        "hh_quarantine_only": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": False,
        },
        "manual_tracing_work_only": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            # "met_before_w": 1.,
            # "met_before_o": 1.,
            "manual_othr_trace_prob": 0.0,
        },
        "manual_tracing": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            # "met_before_w": 1.,
            # "met_before_o": 1.
        },
        "manual_tracing_limit_othr_contact": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "max_contacts": 4,
        },
        "manual_tracing_met_all_before": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "met_before_w": 1.0,
            "met_before_o": 1.0,
        },
        "manual_tracing_met_all_before_limit_othr_contact": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": False,
            "do_pop_testing": False,
            "met_before_w": 1.0,
            "met_before_o": 1.0,
            "max_contacts": 4,
        },
        "app_tracing": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": False,
            "do_app_tracing": True,
            "do_pop_testing": False,
            # "met_before_w": 1.,
            # "met_before_o": 1.
        },
        "app_tracing_limit_othr_contact": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": False,
            "do_app_tracing": True,
            "do_pop_testing": False,
            # "met_before_w": 1.,
            # "met_before_o": 1.,
            "max_contacts": 4,
        },
        "both_tracing": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": True,
            "do_pop_testing": False,
            # "met_before_w": 1.,
            # "met_before_o": 1.
        },
        "both_tracing_limit_othr_contact": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": True,
            "do_pop_testing": False,
            # "met_before_w": 1.,
            # "met_before_o": 1.,
            "max_contacts": 4,
        },
        "pop_testing": {
            "do_individual_isolation": True,
            "do_household_isolation": False,
            "do_manual_tracing": False,
            "do_app_tracing": False,
            "do_pop_testing": True,
            "do_symptom_testing": False,
            "p_pop_test": 0.05,
        },
        "all": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": True,
            "do_pop_testing": True,
            # "met_before_w": 1.,
            # "met_before_o": 1.
            "p_pop_test": 0.05,
        },
        "all_met_limit": {
            "do_individual_isolation": True,
            "do_household_isolation": True,
            "do_manual_tracing": True,
            "do_app_tracing": True,
            "do_pop_testing": True,
            # "met_before_w": 1.,
            # "met_before_o": 1.,
            "max_contacts": 4,
            "p_pop_test": 0.05,
        },
    },
    "delve": {
        "S5_no_TTI": {**S_levels["S5"], **contact_trace_options["no_TTI"]},
        "S5_symptom_based_TTI": {
            **S_levels["S5"],
            **contact_trace_options["symptom_based_TTI"],
        },
        "S5_test_based_TTI": {
            **S_levels["S5"],
            **contact_trace_options["test_based_TTI"],
        },
        "S5_test_based_TTI_test_contacts": {
            **S_levels["S5"],
            **contact_trace_options["test_based_TTI_test_contacts"],
        },
        "S4_no_TTI": {**S_levels["S4"], **contact_trace_options["no_TTI"]},
        "S4_symptom_based_TTI": {
            **S_levels["S4"],
            **contact_trace_options["symptom_based_TTI"],
        },
        "S4_test_based_TTI": {
            **S_levels["S4"],
            **contact_trace_options["test_based_TTI"],
        },
        "S4_test_based_TTI_test_contacts": {
            **S_levels["S4"],
            **contact_trace_options["test_based_TTI_test_contacts"],
        },
        "S3_no_TTI": {**S_levels["S3"], **contact_trace_options["no_TTI"]},
        "S3_symptom_based_TTI": {
            **S_levels["S3"],
            **contact_trace_options["symptom_based_TTI"],
        },
        "S3_test_based_TTI": {
            **S_levels["S3"],
            **contact_trace_options["test_based_TTI"],
        },
        "S3_test_based_TTI_test_contacts": {
            **S_levels["S3"],
            **contact_trace_options["test_based_TTI_test_contacts"],
        },
        "S2_no_TTI": {**S_levels["S2"], **contact_trace_options["no_TTI"]},
        "S2_symptom_based_TTI": {
            **S_levels["S2"],
            **contact_trace_options["symptom_based_TTI"],
        },
        "S2_test_based_TTI": {
            **S_levels["S2"],
            **contact_trace_options["test_based_TTI"],
        },
        "S2_test_based_TTI_test_contacts": {
            **S_levels["S2"],
            **contact_trace_options["test_based_TTI_test_contacts"],
        },
        "S1_no_TTI": {**S_levels["S1"], **contact_trace_options["no_TTI"]},
        "S1_symptom_based_TTI": {
            **S_levels["S1"],
            **contact_trace_options["symptom_based_TTI"],
        },
        "S1_test_based_TTI": {
            **S_levels["S1"],
            **contact_trace_options["test_based_TTI"],
        },
        "S1_test_based_TTI_test_contacts": {
            **S_levels["S1"],
            **contact_trace_options["test_based_TTI_test_contacts"],
        },
        "S0": S_levels["S0"],
    },
}

_policy_configs["delve"].update(
    {
        "S5_test_based_TTI_full_compliance": {
            **S_levels["S5"],
            **contact_trace_options["test_based_TTI"],
            "compliance": 1.0,
        },
        "S4_test_based_TTI_full_compliance": {
            **S_levels["S4"],
            **contact_trace_options["test_based_TTI"],
            "compliance": 1.0,
        },
        "S3_test_based_TTI_full_compliance": {
            **S_levels["S3"],
            **contact_trace_options["test_based_TTI"],
            "compliance": 1.0,
        },
        "S2_test_based_TTI_full_compliance": {
            **S_levels["S2"],
            **contact_trace_options["test_based_TTI"],
            "compliance": 1.0,
        },
        "S1_test_based_TTI_full_compliance": {
            **S_levels["S1"],
            **contact_trace_options["test_based_TTI"],
            "compliance": 1.0,
        },
    }
)

_global_defaults = {
    "cmmid": dict(
        do_isolation=False,  # Impose isolation on symptomatic individual
        do_manual_tracing=False,  # Perform manual contact tracing
        do_app_tracing=False,  # Perform app-based contact tracing. ALT - could set phone prop to 0 if not active
        do_pop_testing=False,  # Randomly test a proportion of the population
        do_schools_open=True,  # If schools in the country are open or not
        manual_home_trace_prob=1.0,  # Probability of home contacts traces
        manual_work_trace_prob=0.95,  # Probability of tracing a work contact
        manual_othr_trace_prob=0.95,  # Probability of tracing an other contact
        met_before_w=0.79,  # At work. At school=90%, which is defined in function later on
        met_before_s=0.9,  # At school. Will replace at work for under 18's
        met_before_o=0.52,  # In other settings
        max_contacts=2e3,  # Enforced limit on number of contacts. Default of 200 to represent no limits
        wfh_prob=0,  # Probability people are working from home
        app_cov=0.53,  # App coverage
        p_pop_test=0.05,  # Proportion mass tested (5% per week)
        policy_adherence=0.9,  # Adherence to testing/trace and quarantine
    ),
    "cmmid_better": dict(
        do_individual_isolation=False,  # Impose isolation on symptomatic individual
        do_household_isolation=False,  # Impose isolation on household of symptomatic individual
        do_manual_tracing=False,  # Perform manual contact tracing
        do_app_tracing=False,  # Perform app-based contact tracing. ALT - could set phone prop to 0 if not active
        do_pop_testing=False,  # Randomly test a proportion of the population
        do_symptom_testing=True,
        do_schools_open=True,  # If schools in the country are open or not
        manual_home_trace_prob=1.0,  # Probability of home contacts traces
        manual_work_trace_prob=0.95,  # Probability of tracing a work contact
        manual_othr_trace_prob=0.95,  # Probability of tracing an other contact
        met_before_w=0.79,  # At work. At school=90%, which is defined in function later on
        met_before_s=0.9,  # At school. Will replace at work for under 18's
        met_before_o=0.52,  # In other settings
        max_contacts=2e3,  # Enforced limit on number of contacts. Default of 200 to represent no limits
        wfh_prob=0,  # Probability people are working from home
        app_cov=0.53,  # App coverage
        p_pop_test=0.05,  # Proportion mass tested (5% per week)
        policy_adherence=0.9,  # Adherence to testing/trace and quarantine
    ),
    "delve": dict(
        isolate_individual_on_symptoms=True,  # Isolate the individual after they present with symptoms
        isolate_individual_on_positive=True,  # Isolate the individual after they test positive
        isolate_household_on_symptoms=False,  # Isolate the household after individual present with symptoms
        isolate_household_on_positive=True,  # Isolate the household after individual test positive
        isolate_contacts_on_symptoms=False,  # Isolate the contacts after individual present with symptoms
        isolate_contacts_on_positive=True,  # Isolate the contacts after individual test positive
        test_contacts_on_positive=False,  # Do we test contacts of a positive case immediately, or wait for them to develop symptoms
        do_symptom_testing=True,  # Test symptomatic individuals
        do_manual_tracing=True,  # Perform manual tracing of contacts
        do_app_tracing=True,  # Perform app tracing of contacts
        fractional_infections=True,  # Include infected but traced individuals as a fraction of their infection period not isolated
        testing_delay=2,  # Days delay between test and results
        app_trace_delay=0,  # Delay associated with tracing through the app
        manual_trace_delay=1,  # Delay associated with tracing manually
        manual_home_trace_prob=1.0,  # Probability of manually tracing a home contact
        manual_work_trace_prob=1.0,  # Probability of manually tracing a work contact
        manual_othr_trace_prob=1.0,  # Probability of manually tracing an other contact
        met_before_w=1.0,  # Probability of having met a work contact before to be able to manually trace
        met_before_s=1.0,  # Probability of having met a school contact before to be able to manually trace
        met_before_o=1.0,  # Probability of having met a other contact before to be able to manually trace
        max_contacts=2e3,  # Place a limit on the number of other contacts per day
        quarantine_length=14,  # Length of quarantine imposed on COVID cases (and household)
        latent_period=3,  # Length of a cases incubation period (from infection to start of infectious period)
        # Parameters for CaseFactors simulation
        app_cov=0.35,
        compliance=0.8,  # Probability of a traced contact isolating correctly
        go_to_school_prob=1.0,  # Fraction of school children attending school
        wfh_prob=0.0,  # Proportion or the population working from home
    ),
}


# strategy factors include all parameters except a few
DELVE_STRATEGY_FACTOR_KEYS = tuple(
    k
    for k in _global_defaults["delve"].keys()
    if k not in ("go_to_school_prob", "wfh_prob")
)
DELVE_CASE_FACTOR_KEYS = ("app_cov", "compliance", "go_to_school_prob", "wfh_prob")


_policy_configs = {
    name: {k: dict(_global_defaults[name], **params) for k, params in strat.items()}
    for name, strat in _policy_configs.items()
}


def get_strategy_configs(strategy_name, config_names=None):
    """
    Returns configurations for specified strategy.

    :param strategy_name: Name of the strategy
    :param config_names: List of configurations. Each must be valid for a given strategy.
                         If None, all configurations for a given strategy are returned.
    """
    try:
        strategy = _policy_configs[strategy_name.lower()]
    except KeyError:
        raise ValueError(f"Cannot find strategy {strategy_name} in config.py")

    if config_names is None:
        return dict(**strategy)

    # to avoid confusing errors when string is passed
    if isinstance(config_names, str):
        config_names = [config_names]

    output = dict()
    for config_name in config_names:
        try:
            output[config_name] = dict(**strategy[config_name])
        except KeyError:
            raise ValueError(
                f"Cannot find configuration {config_name} under "
                f"strategy {strategy_name} in config.py"
            )
    return output


Sensitivity = namedtuple("Sensitivity", ["bounds", "values"])


_policy_sensitivities = {
    "delve": dict(
        app_cov=Sensitivity(bounds=(0, 1), values=np.linspace(0.1, 1.0, num=10)),
        testing_delay=Sensitivity(bounds=None, values=[1, 2, 3]),
        latent_period=Sensitivity(bounds=None, values=[2, 3]),
        manual_trace_delay=Sensitivity(bounds=None, values=[1, 2, 3]),
        compliance=Sensitivity(bounds=None, values=np.linspace(0.5, 1.0, 6)),
    )
}

# symp covid neg, symp covid pos, asymp covid pos
# Covid+ individuals: 10k, 20k [default], 30k
# flu-like symptoms (non-covid): 50k, 100k [default], 200k, 300k
_vary_flu = [
    {
        "dist": [
            k / (k + 20),
            PROP_COVID_SYMPTOMATIC * 20 / (k + 20),
            (1 - PROP_COVID_SYMPTOMATIC) * 20 / (k + 20),
        ],
        "nppl": k + 20,
    }
    for k in [50, 100, 200, 300]
]

_vary_covid = [
    {
        "dist": [
            100 / (100 + k),
            PROP_COVID_SYMPTOMATIC * k / (100 + k),
            (1 - PROP_COVID_SYMPTOMATIC) * k / (100 + k),
        ],
        "nppl": k + 100,
    }
    for k in [10, 20, 30]
]

_inf_prop_to_try = _vary_flu
_inf_prop_to_try.extend(_vary_covid)
_inf_prop_to_try.append(
    {
        "dist": [
            100 / (120),
            PROP_COVID_SYMPTOMATIC * 20 / (120),
            (1 - PROP_COVID_SYMPTOMATIC) * 20 / (120),
        ],
        "nppl": 120,
    }
)

_case_sensitivities = {
    # to be decided!
    "delve": dict(
        infection_proportions=Sensitivity(
            bounds=None,
            values=_inf_prop_to_try
            # # symp covid neg, symp covid pos, asymp covid pos
            # [200/260, 0.6 * 60/260, 0.4 * 60/260],  # anne
            # [150/(180+30), 0.6 * 60/(180+30), 0.4 * 60/(180+30)],  # Bugwatch May
            # [150/(100+30), 0.6 * 60/(100+30), 0.4 * 60/(100+30)],  # Bugwatch June
            # ],
        ),
        p_day_noticed_symptoms=Sensitivity(
            bounds=None,
            values=[
                [
                    0.0,
                    0.0,
                    0.25,
                    0.25,
                    0.2,
                    0.1,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                ],  # pessimistic, mean delay 4.05 days
                [
                    0,
                    0.25,
                    0.25,
                    0.2,
                    0.1,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.00,
                ],  # mean delay 3.05 days
                [
                    0.0,
                    0.5,
                    0.2,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # optimistic, mean delay 2.1 days
            ],
        ),
        inf_profile=Sensitivity(
            bounds=None,
            values=[
                (
                    he_infection_profile(  # pessimistic from He et al.
                        period=10, gamma_params={"a": 2.11, "scale": 1 / 0.69}
                    )
                ).tolist(),
                (
                    he_infection_profile(  # pessimistic from He et al.
                        period=10, gamma_params={"a": 2.80, "scale": 1 / 0.69}
                    )
                ).tolist(),
                (
                    he_infection_profile(  # pessimistic from He et al.
                        period=10, gamma_params={"a": 3.49, "scale": 1 / 0.69}
                    )
                ).tolist(),
            ],
        ),
    )
}


get_strategy_sensitivities = partial(
    get_contacts_config, _cfg_dct=_policy_sensitivities
)
get_case_sensitivities = partial(get_contacts_config, _cfg_dct=_case_sensitivities)


if __name__ == "__main__":
    for k, v in _policy_sensitivities.items():
        assert k in _policy_configs

    for k, v in _case_sensitivities.items():
        assert k in _case_configs

    for prop in _inf_prop_to_try:
        assert sum(prop["dist"]) == 1, f"{prop['dist']}, sums to {sum(prop['dist'])}"
