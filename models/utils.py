import numpy as np
from scipy.stats import gamma


def bool_bernoulli(p, rng):
    return bool(rng.binomial(1, p))


def categorical(pvals, rng):
    return np.argwhere(rng.multinomial(1, pvals)).item()


def he_infection_profile(period, gamma_params):
    inf_days = np.arange(period)

    mass = gamma.cdf(inf_days + 1, **gamma_params) - gamma.cdf(inf_days, **gamma_params)
    return mass / np.sum(mass)


class Registry:
    "Case insensitive registry"
    def __init__(self):
        self._register = dict()
    
    def __getitem__(self, key):
        return self._register[key.lower()]
    
    def __call__(self, name):
        def add(thing):
            self._register[name.lower()] = thing
            return thing
        return add
