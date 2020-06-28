import numpy as np
import pytest

from tti_explorer.case_statistics import CaseStatistics
from tti_explorer.case import Case
from tti_explorer.contacts import Contacts


@pytest.fixture
def covid_case():
    return Case(
        covid=True,
        symptomatic=True,
        under18=False,
        day_noticed_symptoms=1,
        inf_profile=[0.5, 0.5],
    )

@pytest.fixture
def non_covid_case():
    return Case(
        covid=False,
        symptomatic=False,
        under18=False,
        day_noticed_symptoms=-1,
        inf_profile=[0.5, 0.5],
    )


def test_covid_case_statistics(covid_case):
    contacts = Contacts(1, np.ones((1,2), dtype=int), np.ones((1,2), dtype=int), np.ones((1,2), dtype=int))

    stats = CaseStatistics([(covid_case, contacts)])

    assert stats.covid_count == 1
    assert stats.case_count == 1
    assert (stats.mean_R == np.array([1, 1, 1, 3])).all()
    assert (stats.std_R == 0).all()


def test_mix_cases_statistics(covid_case, non_covid_case):
    contacts = Contacts(1, np.ones((1,2), dtype=int), np.ones((1,2), dtype=int), np.ones((1,2), dtype=int))

    stats = CaseStatistics([(covid_case, contacts), (non_covid_case, contacts)])

    assert stats.covid_count == 1
    assert stats.case_count == 2
    assert (stats.mean_R == np.array([1, 1, 1, 3])).all()
    assert (stats.std_R == 0).all()
