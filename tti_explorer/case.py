from dataclasses import dataclass

import numpy as np

from .utils import bool_bernoulli, categorical


@dataclass
class Case:
    under18: bool
    covid: bool
    symptomatic: bool
    day_noticed_symptoms: int
    inf_profile: list

    def to_dict(self):
        return dict(
            under18=self.under18,
            covid=self.covid,
            symptomatic=self.symptomatic,
            day_noticed_symptoms=self.day_noticed_symptoms,
            inf_profile=self.inf_profile,
        )


@dataclass
class CaseFactors:
    wfh: bool
    has_app: bool
    report_app: bool
    report_manual: bool

    @classmethod
    def simulate_from(cls, rng, case, app_cov, go_to_school_prob, wfh_prob, compliance):
        """Simulate case factors

        Args:
            rng:
            case:
            app_cov:
            go_to_school_prob:
            wfh_prob:
            compliance:

        Returns:
        """
        p = (1 - go_to_school_prob) if case.under18 else wfh_prob
        wfh = bool_bernoulli(p, rng)

        has_app = bool_bernoulli(app_cov, rng)
        does_report = case.symptomatic and bool_bernoulli(compliance, rng)
        report_app = does_report and has_app
        report_manual = does_report and not has_app

        return cls(
            wfh=wfh, has_app=has_app, report_app=report_app, report_manual=report_manual
        )


def simulate_case(
    rng, p_under18, infection_proportions, p_day_noticed_symptoms, inf_profile
):
    """simulate_case

    Args:
        rng (np.random.RandomState): random number generator.
        p_under18 (float): Probability of case being under 18
        infection_proportions (dict): Probs of being symp covid neg, symp covid pos, asymp covid pos
                                      The only required key is 'dist', which contains list of the named probs, in that order.
        p_day_noticed_symptoms (np.array[float]): Distribution of day on which case notices
            their symptoms. (In our model this is same as reporting symptoms.)
            Conditional on being symptomatic.
        inf_profile (list[float]): Distribution of initial exposure of positive secondary cases
            relative to start of primary case's infectious period.

    Returns (Case): case with attributes populated.
    """
    (
        p_symptomatic_covid_neg,
        p_symptomatic_covid_pos,
        p_asymptomatic_covid_pos,
    ) = infection_proportions["dist"]

    under18 = bool_bernoulli(p_under18, rng)

    illness_pvals = [
        p_asymptomatic_covid_pos,
        p_symptomatic_covid_neg,
        p_symptomatic_covid_pos,
    ]
    illness = categorical(illness_pvals, rng).item()

    if illness == 0:
        return Case(
            covid=True,
            symptomatic=False,
            under18=under18,
            day_noticed_symptoms=-1,
            inf_profile=np.array(inf_profile),
        )
    else:
        covid = illness == 2
        profile = np.array(inf_profile) if covid else np.zeros(len(inf_profile))
        return Case(
            covid=covid,
            symptomatic=True,
            under18=under18,
            day_noticed_symptoms=categorical(p_day_noticed_symptoms, rng).item(),
            inf_profile=profile,
        )
