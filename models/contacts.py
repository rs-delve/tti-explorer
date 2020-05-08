from collections import namedtuple

import numpy as np

import utils

NOT_INFECTED = -1
NCOLS = 2


def day_infected_wo(rng, probs, first_encounter, not_infected=NOT_INFECTED):
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
            home_inf_profile = utils.home_daily_infectivity(case.inf_profile)
            day_infected = utils.categorical(home_inf_profile, rng=self.rng, n=n_home)
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


if __name__ == "__main__":
    # Basic testing
    import os
    from types import SimpleNamespace

    from generate_cases import simulate_case
    from config import get_case_config

    data_folder = "../data"

    home_sar = work_sar = other_sar = 0.2
    period = 5

    rng = np.random.RandomState(0)

    def load_csv(pth):
        return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")

    over18 = load_csv(os.path.join(data_folder, "contact_distributions_o18.csv"))
    under18 = load_csv(os.path.join(data_folder, "contact_distributions_u18.csv"))

    contact_simluator = EmpiricalContactsSimulator(over18, under18, rng)

    for _ in range(10):
        case = simulate_case(rng, **get_case_config("oxteam"))
        print(contact_simluator(case, 0.2, 0.03, 0.03, 0.5, 10))
