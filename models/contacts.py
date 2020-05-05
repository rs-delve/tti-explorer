import itertools

from collections import namedtuple

import numpy as np


NOT_INFECTED = -1


def home_daily_infectivity(base_risk, infective_days):
    # Compute daily risk such that the probability of a binomial sample of (infective_days, daily risk)
    # has p(x=0) = 1 - base_risk (i.e. p(x>0) = base_risk) 
    return 1 - np.power((1-base_risk), 1./float(infective_days))


def get_day_infected_home(mat, not_infected=NOT_INFECTED):
    return np.where(mat.any(axis=1), mat.argmax(axis=1), not_infected)


def get_day_infected_wo(mat, period, n_ppl, not_infected=NOT_INFECTED):
    return np.where(
        mat,
        np.repeat(np.arange(period), n_ppl),
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

    def __call__(self, case, home_sar, work_sar, other_sar, period):
        row = self.sample_row(case)
        n_home, n_work, n_other = row

        infectivity_propn = 1.0 if case.symptomatic else 0.5

        # From the overall home SAR compute the daily infection chance
        home_daily_sar = home_daily_infectivity(home_sar * infectivity_propn, period)

        home_is_infected = self.rng.binomial(1, home_daily_sar, size=(n_home, period))
        work_is_infected = self.rng.binomial(1, work_sar * infectivity_propn, size=n_work * period)
        other_is_infected = self.rng.binomial(1, other_sar * infectivity_propn, size=n_other * period)

        home_inf = get_day_infected_home(home_is_infected)
        work_inf = get_day_infected_wo(work_is_infected, period, n_work)
        other_inf = get_day_infected_wo(other_is_infected, period, n_other)

        home_first_encounter = np.zeros(n_home, dtype=int)
        work_first_encounter = np.repeat(np.arange(period), n_work)
        other_first_encounter = np.repeat(np.arange(period), n_other)
        return Contacts(
                n_daily=dict(zip("home work other".split(), row)),
                home=np.column_stack((home_inf, home_first_encounter)),
                work=np.column_stack((work_inf, work_first_encounter)),
                other=np.column_stack((other_inf, other_first_encounter))
            )


if __name__ == "__main__":
    # Basic testing
    import os 
    from types import SimpleNamespace

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
        over18 = np.random.binomial(n=1, p=0.21)
        case = SimpleNamespace(over18=over18)

        n_home, n_work, n_other = contact_simluator.sample_row(case)

        home_is_infected = rng.binomial(1, home_sar, size=(n_home, period))
        work_is_infected = rng.binomial(1, work_sar, size=n_work * period)
        other_is_infected = rng.binomial(1, other_sar, size=n_other * period)

        home_inf = get_day_infected_home(home_is_infected)
        work_inf = get_day_infected_wo(work_is_infected, period, n_work)
        other_inf = get_day_infected_wo(other_is_infected, period, n_other)

        home_first_encounter = np.zeros(n_home)
        work_first_encounter = np.repeat(np.arange(period), n_work)
        other_first_encounter = np.repeat(np.arange(period), n_other)

        contacts = contact_simluator(case, home_sar, work_sar, other_sar, period)
        print(contacts)
