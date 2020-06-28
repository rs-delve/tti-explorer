import numpy as np


def _count_infected(contacts):
    return np.sum(contacts[:, 0] >= 0)


class CaseStatistics():
    def __init__(self, cases_contacts):
        n_covid = 0
        stats = []
        for case, contacts in cases_contacts:
            if case.covid:
                n_covid += 1
                home = _count_infected(contacts.home)
                work = _count_infected(contacts.work)
                other = _count_infected(contacts.other)
                total = home + work + other
                stats.append((home, work, other, total))

        self.n_covid = n_covid
        self.n_cases = len(cases_contacts)
        self.stats = np.array(stats)

    @property
    def covid_count(self):
        return self.n_covid

    @property
    def case_count(self):
        return self.n_cases

    @property
    def mean_R(self):
        return self.stats.mean(0)

    @property
    def std_R(self):
        return self.stats.std(0)
