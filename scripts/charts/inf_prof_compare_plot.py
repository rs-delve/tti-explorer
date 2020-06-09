

import matplotlib.pyplot as plt

import numpy as np

from tti_explorer.contacts import home_daily_infectivity, he_infection_profile

loc = 0
# taken from (ref)
gamma_params = {
    'a': 2.80,
    'loc': loc,
    'scale': 1/0.69
}
t = 10
days = np.arange(t)
mass = he_infection_profile(t, gamma_params)

plt.style.use('default')
fig, ax = plt.subplots(1, figsize=(9*0.8, 5*0.8))
xaxis = np.linspace(-2, t, 1000)
skewed_mass = home_daily_infectivity(mass)
ax.bar(days + 0.1, skewed_mass, label="Home", align="edge", color="C1", zorder=0)
ax.bar(days, mass, label="Work/other", align="edge", zorder=1)
ax.plot(np.arange(6), [1/5, 1/5, 1/5, 1/5, 1/5, 1/5],
        label="Kucharski et al.", ls='--', color="k", alpha=0.8, zorder=2)

ax.legend(loc="upper right")
ax.set_axis_on()
ax.set_ylabel('Distribution of initial infection by contact type ')
ax.set_xlabel('Days since start of infectious period')
ax.set_xticks(np.arange(t + 1))
plt.show()
# fig.savefig('./inf_profile.pdf')
