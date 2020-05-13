from collections import namedtuple

import numpy as np
from scipy.stats import gamma

from .utils import categorical

NOT_INFECTED = -1
NCOLS = 2


def he_infection_profile(period, gamma_params):
    """he_infection_profile

    Args:
        period (int): length of infectious period
        gamma_params (dict): shape and scale gamma parameters of infection profile

    Returns: discretised and truncated gamma cdf, modelling the infection profile
    """
    inf_days = np.arange(period)
    mass = gamma.cdf(inf_days + 1, **gamma_params) - gamma.cdf(inf_days, **gamma_params)
    return mass / np.sum(mass)


def home_daily_infectivity(base_mass):
    """home_daily_infectivity

    Args:
        base_mass:

    Returns:
    """
    fail_prod = np.cumprod(1 - base_mass)
    fail_prod = np.roll(fail_prod, 1)
    np.put(fail_prod, 0, 1.)
    skewed_mass = fail_prod * base_mass
    return skewed_mass / np.sum(skewed_mass)


def day_infected_wo(rng, probs, first_encounter, not_infected=NOT_INFECTED):
    """day_infected_wo

    Args:
        rng (np.random.RandomState):
        probs:
        first_encounter:
        not_infected:

    Returns:
    """
    return np.where(
        rng.binomial(n=1, p=probs),
        first_encounter,
        not_infected
    )


Contacts = namedtuple(
        'Contacts',
        ['n_daily', 'home', 'work', 'other']
    )


class EmpiricalContactsSimulator:
    """Simulate social contact using BBC Pandemic data"""
    def __init__(self, over18, under18, rng):
        self.over18 = over18
        self.under18 = under18
        self.rng = rng

    def sample_row(self, case):
        table = self.under18 if case.under18 else self.over18
        return table[self.rng.randint(0, table.shape[0])]

    def __call__(self, case, home_sar, work_sar, other_sar, asymp_factor, period):
        """__call__

        Args:
            case:
            home_sar:
            work_sar:
            other_sar:
            asymp_factor:
            period:

        Returns:
        """
        row = self.sample_row(case)
        n_home, n_work, n_other = row

        scale = 1.0 if case.symptomatic else asymp_factor

        home_first_encounter = np.zeros(n_home, dtype=int)
        work_first_encounter = np.repeat(np.arange(period, dtype=int), n_work)
        other_first_encounter = np.repeat(np.arange(period, dtype=int), n_other)

        if case.covid:
            home_is_infected = self.rng.binomial(1, scale * home_sar, n_home)
            home_inf_profile = home_daily_infectivity(case.inf_profile)
            day_infected = categorical(home_inf_profile, rng=self.rng, n=n_home)
            home_day_inf = np.where(home_is_infected, day_infected, NOT_INFECTED)

            work_day_inf = day_infected_wo(
                    self.rng,
                    probs=work_sar * scale * period * case.inf_profile[work_first_encounter],
                    first_encounter=work_first_encounter,
                    not_infected=NOT_INFECTED
                )

            other_day_inf = day_infected_wo(
                    self.rng,
                    probs=other_sar * scale * period * case.inf_profile[other_first_encounter],
                    first_encounter=other_first_encounter,
                    not_infected=NOT_INFECTED
                )
        else:
            home_day_inf = np.full_like(home_first_encounter, -1)
            work_day_inf = np.full_like(work_first_encounter, -1)
            other_day_inf = np.full_like(other_first_encounter, -1)

        return Contacts(
                n_daily=dict(zip("home work other".split(), row)),
                home=np.column_stack((home_day_inf, home_first_encounter)),
                work=np.column_stack((work_day_inf, work_first_encounter)),
                other=np.column_stack((other_day_inf, other_first_encounter))
            )
