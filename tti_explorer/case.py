from collections import namedtuple

import numpy as np

from .utils import bool_bernoulli, categorical


Case = namedtuple(
        'Case',
        [
            "under18",
            "covid",
            "symptomatic",
            "day_noticed_symptoms",
            "inf_profile"
        ]
    )


def simulate_case(rng, p_under18, infection_proportions, p_day_noticed_symptoms, inf_profile):
    """simulate_case

    Args:
        rng (np.random.RandomState): random number generator.
        p_under18 (float): Probability of case being under 18
        infection_proportions (list[float]): Probs of being symp covid neg, symp covid pos, asymp covid pos
        p_day_noticed_symptoms (np.array[float]): Distribution of day on which case notices
            their symptoms. (In our model this is same as reporting symptoms.)
            Conditional on being symptomatic.
        inf_profile (list[float]): Distribution of initial exposure of positive secondary cases
            relative to start of primary case's infectious period.

    Returns (Case): case with attributes populated.
    """
    p_symptomatic_covid_neg, p_symptomatic_covid_pos, p_asymptomatic_covid_pos = infection_proportions['dist']

    under18 = bool_bernoulli(p_under18, rng)

    illness_pvals = [
                p_asymptomatic_covid_pos,
                p_symptomatic_covid_neg,
                p_symptomatic_covid_pos,
            ]
    illness = categorical(illness_pvals, rng)

    if illness == 0:
        return Case(
                covid=True,
                symptomatic=False,
                under18=under18,
                day_noticed_symptoms=-1,
                inf_profile=np.array(inf_profile)
            )
    else:
        covid = illness == 2
        profile = np.array(inf_profile) if covid else np.zeros(len(inf_profile))
        return Case(
                covid=covid,
                symptomatic=True,
                under18=under18,
                day_noticed_symptoms=categorical(p_day_noticed_symptoms, rng),
                inf_profile=profile
            )
