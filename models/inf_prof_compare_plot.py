

import matplotlib.pyplot as plt

from scipy.stats import gamma

import numpy as np

def home_daily_infectivity(base_mass):
    fail_prod = np.cumprod(1 - base_mass)
    fail_prod = np.roll(fail_prod, 1)
    np.put(fail_prod, 0, 1.)
    skewed_mass = fail_prod * base_mass
    return skewed_mass / np.sum(skewed_mass)

loc =  0
# taken from (ref)
gamma_params = {
    'a': 2.80,
    'loc': loc,
    'scale': 1/0.69
}
t = 10
days = np.arange(t)
mass = gamma.cdf(days + 1, **gamma_params) - gamma.cdf(days, **gamma_params)
mass = mass / np.sum(mass)
plt.style.use('default')
fig, ax = plt.subplots(1, figsize=(9*0.8, 5*0.8))
xaxis = np.linspace(-2, t, 1000)
skewed_mass = home_daily_infectivity(mass)
ax.bar(days + 0.1, skewed_mass, label="Home", align="edge", color = "C1", zorder = 0)
ax.bar(days, mass, label="Work/other", align="edge", zorder = 1)
ax.plot(np.arange(6), [1/5, 1/5, 1/5, 1/5, 1/5, 1/5], label="Kucharski et al.", ls = '--', color = "k", alpha = 0.8, zorder = 2) #ls="-", color="C1", lw=5, zorder = 0)

# ax.plot(xaxis, gamma.pdf(xaxis, **gamma_params), color="C1", zorder=10, label="PDF")
ax.legend(loc="upper right")
# ax.set_xlim(-2, 5)
# ax.set_ylim(-0.1, 0.5)
ax.set_axis_on()
ax.set_ylabel('Distribution of initial infection by contact type ')
ax.set_xlabel('Days since start of infectious period')
ax.set_xticks(np.arange(t + 1))
# plt.show()
fig.savefig('./inf_profile.pdf')
