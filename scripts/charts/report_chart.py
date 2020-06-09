
import matplotlib.pyplot as plt
import numpy as np

from tti_explorer.contacts import he_infection_profile

plt.style.use('default')
loc = 0
# taken from He et al
gamma_params = {
    'a': 2.11,
    'loc': loc,
    'scale': 1/0.69
}
t = 10
days = np.arange(t)

mass = he_infection_profile(t, gamma_params)

fig, ax = plt.subplots(1, figsize=(9*0.8, 5*0.8))
xaxis = np.linspace(-2, t, 1000)
ax.bar(
    np.arange(5)+0.1,
    [1/5, 1/5, 1/5, 1/5, 1/5],
    label="Kucharski profile",
    align="edge",
    color="C1",
    zorder=1,
    alpha=0.6
)
ax.bar(days, mass, label="Discretised", align="edge", zorder=1)
ax.legend(loc="upper right")
ax.set_axis_on()
ax.set_ylabel('Secondary attack profile')
ax.set_xlabel('Days since start of infectious period')
ax.set_xticks(days)
plt.show()
# fig.savefig('./charts/inf_profile.pdf')
