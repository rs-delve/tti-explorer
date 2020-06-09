from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma

from .utils import categorical

NOT_INFECTED = -1
NCOLS = 2


@dataclass
class Contacts:
    n_daily: dict
    home: np.array
    work: np.array
    other: np.array

    def to_dict(self):
        return dict(
            n_daily=self.n_daily, home=self.home, work=self.work, other=self.other
        )


def he_infection_profile(period, gamma_params):
    """he_infection_profile

    Args:
        period (int): length of infectious period
        gamma_params (dict): shape and scale gamma parameters
        of infection profile

    Returns:
        infection_profile (np.array[float]): discretised and
        truncated gamma cdf, modelling the infection profile
        over period
    """
    inf_days = np.arange(period)
    mass = gamma.cdf(inf_days + 1, **gamma_params) - gamma.cdf(inf_days, **gamma_params)
    return mass / np.sum(mass)


def home_daily_infectivity(base_mass):
    """home_daily_infectivity

    Args:
        base_mass (np.array[float]): infection profile for
        non-repeat contacts

    Returns:
        infection_profile (np.array[float]):
        infection profile for repeat contacts
    """
    fail_prod = np.cumprod(1 - base_mass)
    fail_prod = np.roll(fail_prod, 1)
    np.put(fail_prod, 0, 1.0)
    skewed_mass = fail_prod * base_mass
    return skewed_mass / np.sum(skewed_mass)


def day_infected_wo(rng, probs, first_encounter, not_infected=NOT_INFECTED):
    """day_infected_wo

    Args:
        rng (np.random.RandomState): Random state.
        probs (np.array[float]): Probability of infection of contact each.
        first_encounter (np.array[float]): Day of first encounter of contact with
        primary case.
        not_infected (float): Flag to use if the contact was not infected.

    Returns:
        day_infected (np.array[int]): The day on which the contacts were infected,
        if not infected then the element for that contact will be NOT_INFECTED.
    """
    return np.where(rng.binomial(n=1, p=probs), first_encounter, not_infected)


class EmpiricalContactsSimulator:
    """Simulate social contact using BBC Pandemic data"""

    def __init__(self, over18, under18, rng):
        """Simulate social contact using the BBC Pandemic dataset

            Each row in input arrays consists of three numbers,
            represeting number of contacts at: home, work, other

        Args:
            over18 (np.array[int], Nx3): Contact data for over 18s.
            under18 (np.array[int], Nx3): Contact data for under 18s.
            rng (np.random.RandomState): Random state.

        """
        self.over18 = over18
        self.under18 = under18
        self.rng = rng

    def sample_row(self, case):
        """sample_row
        Sample a row of the tables depending on the age of the case.

        Args:
            case (Case): Primary case.

        Returns:
            row (np.array[int]): Row sampled uniformly at random from table in
            dataset depending on age of case (over/under18). Three columns,
            expected contacts for categories home, work and other.
            For under 18s, school contacts are interpreted as work contacts.
        """
        table = self.under18 if case.under18 else self.over18
        return table[self.rng.randint(0, table.shape[0])]

    def __call__(self, case, home_sar, work_sar, other_sar, asymp_factor, period):
        """Generate a social contact for the given case.

        A row from the table corresponding to the age of the `case` is sampled
        uniformly at random. A contact is generated with daily contacts as
        given by that row. These contacts are infected at random with attack rates
        given by the SARs and whether or not the `case` is symptomatic. If the
        `case` is COVID negative, then no contacts are infected.

        Args:
            case (Case): Primary case.
            home_sar (float): Secondary attack rate for household contacts.
                              (Marginal probability of infection over the whole simulation)
            work_sar (float): Secondary attack rate for contacts in the work category.
            other_sar (float): Secondary attack rate for contacts in the other category.
            asymp_factor (float): Factor by which to multiply the probabilty of secondary
                                  infection if `case` is asymptomatic COVID positive.
            period (int): Duration of the simulation (days).

        Returns:
            contacts (Contacts): Simulated social contacts and resulting infections
            for primary case `case`.
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
                probs=work_sar
                * scale
                * period
                * case.inf_profile[work_first_encounter],
                first_encounter=work_first_encounter,
                not_infected=NOT_INFECTED,
            )

            other_day_inf = day_infected_wo(
                self.rng,
                probs=other_sar
                * scale
                * period
                * case.inf_profile[other_first_encounter],
                first_encounter=other_first_encounter,
                not_infected=NOT_INFECTED,
            )
        else:
            home_day_inf = np.full_like(home_first_encounter, -1)
            work_day_inf = np.full_like(work_first_encounter, -1)
            other_day_inf = np.full_like(other_first_encounter, -1)

        return Contacts(
            n_daily=dict(zip("home work other".split(), row)),
            home=np.column_stack((home_day_inf, home_first_encounter)),
            work=np.column_stack((work_day_inf, work_first_encounter)),
            other=np.column_stack((other_day_inf, other_first_encounter)),
        )
