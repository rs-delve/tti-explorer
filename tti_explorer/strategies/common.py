from types import SimpleNamespace

import numpy as np


# Changing the string values here saves having to change them
# in every strategy and will keep tables consistent
RETURN_KEYS = SimpleNamespace(
    base_r="Base R",
    reduced_r="Effective R",
    man_trace="# Manual Traces",
    app_trace="# App Traces",
    tests="# Tests Needed",
    quarantine="# PersonDays Quarantined",
    covid="Has Covid",
    symptomatic="Is Symptomatic",
    tested="Got tested",
    secondary_infections="# Secondary Infections",
    cases_prevented_social_distancing="# Secondary Infections Prevented by Social Distancing",
    cases_prevented_symptom_isolating="# Secondary Infections Prevented by Isolating Cases with Symptoms",
    cases_prevented_contact_tracing="# Secondary Infections Prevented by Contact Tracing",
    fractional_r="Fractional R",
    # All for cases reporting symptoms only.
    stopped_by_social_distancing_percentage="% of Ongoing Transmission Prevented by Social Distancing",
    stopped_by_symptom_isolating_percentage="% of Ongoing Transmission Prevented by Isolating Cases with Symptoms and Quarantining Households",
    stopped_by_tracing_percentage="% of Ongoing Transmission Prevented by Tracing",
    not_stopped_asymptomatic_percentage="% of Ongoing Transmission Allowed Through by Asymptomatic Cases Not Being Caught",
    not_stopped_symptomatic_non_compliant_percentage="% of Ongoing Transmission Allowed Through by Symptomatic Cases Not Complying",
    not_stopped_by_tti_percentage="% of Ongoing Transmission Allowed Through by TTI Policy",
    stopped_by_social_distancing_symptomatic_compliant_percentage="% of Ongoing Transmission Prevented by Social Distancing - Symptomatic Compliant Only",
    percent_primary_symptomatic_missed="% Primary Symptomatic Cases Missed",
    percent_primary_asymptomatic_missed="% Primary Asymptomatic Cases Missed",
    percent_primary_missed="% Primary Cases Missed",
    percent_secondary_from_symptomatic_missed="% Secondary Cases from Symptomatic Cases Missed",
    percent_secondary_from_asymptomatic_missed="% Secondary Cases from Asymptomatic Cases Missed",
    percent_secondary_missed="% Secondary Cases Cases Missed",
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
    for day in range(np.max(contacts) + 1):
        is_day = contacts == day
        n_on_day = is_day.cumsum()
        allow_on_day = (n_on_day <= max_per_day) & (n_on_day != 0)
        contact_limited = contact_limited | allow_on_day

    return contact_limited
