import numpy as np
import pytest

from tti_explorer.scenario import get_monte_carlo_factors


def test_get_monte_carlo_factors():
    monte_carlo_factor, r_monte_carlo_factor = get_monte_carlo_factors(1, 0.125, 0.125)

    assert monte_carlo_factor == 1.0
    assert r_monte_carlo_factor == 2.0