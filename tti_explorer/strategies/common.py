from types import SimpleNamespace

import numpy as np


# Changing the string values here saves having to change them
# in every strategy and will keep tables consistent
RETURN_KEYS = SimpleNamespace(
        base_r="Base R",
        reduced_r='Effective R',
        man_trace='# Manual Traces',
        app_trace='# App Traces',
        tests='# Tests Needed',
        quarantine='# PersonDays Quarantined',
        wasted_quarantine='# Wasted PersonDays Quarantined'
    )


# BE: this type of masking might be useful to limit contacts
# for home contacts n_days would be 1
def _limit_contact_mask(n_daily, n_days, max_per_day):
    return np.repeat(np.arange(1, n_daily + 1), n_days) <= max_per_day


def _limit_contact(contacts, max_per_day):
    """Generates a boolean array describing if a contact would have 
    been contacted due daily contact limiting.

    Parameters
    ----------
    contacts : 1d array of contact days
    max_per_day : Max contacts per day
    """
    if contacts.size == 0:
        return np.array([]).astype(bool)
    contact_limited = np.zeros_like(contacts).astype(bool)
    for day in range(np.max(contacts)+1):
        is_day = (contacts == day)
        n_on_day = is_day.cumsum()
        allow_on_day = (n_on_day <= max_per_day) & (n_on_day != 0)
        contact_limited = (contact_limited | allow_on_day)

    return contact_limited
