

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
# k_days = np.arange(6)
ax.bar(np.arange(5)+0.1, [1/5, 1/5, 1/5, 1/5, 1/5], label="Kucharski profile", align = "edge", color = "C1", zorder = 1, alpha = 0.6) #ls="-", color="C1", lw=5, zorder = 0)
ax.bar(days, mass, label="Discretised", align="edge", zorder = 1)
# ax.plot(xaxis, gamma.pdf(xaxis, **gamma_params), color="C1", zorder=10, label="PDF")
ax.legend(loc="upper right")
# ax.set_xlim(-2, 5)
# ax.set_ylim(-0.1, 0.5)
ax.set_axis_on()
ax.set_ylabel('Secondary attack profile')
ax.set_xlabel('Days since start of infectious period')
ax.set_xticks(days)
# plt.show()
fig.savefig('./charts/inf_profile.pdf')
