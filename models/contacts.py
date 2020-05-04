import itertools

from collections import namedtuple

import numpy as np

NOT_INFECTED = -1


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
        table = self.over18 if case.over18 else self.under18
        return table[self.rng.randint(0, table.shape[0])]

    def __call__(self, case, home_sar, work_sar, other_sar, period):
        row = self.sample_row(case)
        home, work, other = row

        home_is_infected = self.rng.binomial(1, home_sar, size=(home, period))
        work_is_infected = self.rng.binomial(1, work_sar, size=work * period)
        other_is_infected = self.rng.binomial(1, other_sar, size=other * period)
        return Contacts(
                n_daily=row,
                home=get_day_infected_home(home_is_infected),
                work=get_day_infected_wo(work_is_infected, period, work),
                other=get_day_infected_wo(other_is_infected, period, other)
            )


if __name__ == "__main__":
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
        over18 = np.random.choice([0, 1])
        case = SimpleNamespace(over18=over18)
        contacts = contact_simluator(case, home_sar, work_sar, other_sar, period)
        print(contacts)
