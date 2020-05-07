

import matplotlib.pyplot as plt

from scipy.stats import gamma

import numpy as np

loc =  0
# taken from (ref)
gamma_params = {
    'a': 2.11,
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
ax.bar(days+1, mass, label="Discretised", align="edge")
ax.plot(np.arange(5) + 1, [1/5, 1/5, 1/5, 1/5, 1/5], label="Kucharski profile", ls="--", color="C1", lw=5)
# ax.plot(xaxis, gamma.pdf(xaxis, **gamma_params), color="C1", zorder=10, label="PDF")
ax.legend(loc="upper right")
# ax.set_xlim(-2, 5)
# ax.set_ylim(-0.1, 0.5)
ax.set_axis_on()
ax.set_ylabel('Secondary attack profile')
ax.set_xlabel('Simulation days')
ax.set_xticks(days+1)
plt.show()
fig.savefig('./inf_profile.pdf')

