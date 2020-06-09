import numpy as np

from .. import config, utils
from ..case import CaseFactors
from . import registry
from .common import _limit_contact, RETURN_KEYS


@registry("delve")
def delve(case, contacts, rng, **kwds):
    strategy_factors = utils.get_sub_dictionary(kwds, config.DELVE_STRATEGY_FACTOR_KEYS)
    strategy = TTIFlowModel(rng, **strategy_factors)

    factor_cfg = utils.get_sub_dictionary(kwds, config.DELVE_CASE_FACTOR_KEYS)
    case_factors = CaseFactors.simulate_from(rng, case, **factor_cfg)
    metrics = strategy(case, contacts, case_factors)

    return metrics


# TODO: turn all these comments into (google style) docstring
class TTIFlowModel:
    """
    This is an implementation of flowchart produced by Anne Johnson and Guy Harling
    """

    def __init__(
        self,
        rng,
        isolate_individual_on_symptoms,  # Isolate the individual after they present with symptoms
        isolate_individual_on_positive,  # Isolate the individual after they test positive
        isolate_household_on_symptoms,  # Isolate the household after individual present with symptoms
        isolate_household_on_positive,  # Isolate the household after individual test positive
        isolate_contacts_on_symptoms,  # Isolate the contacts after individual present with symptoms
        isolate_contacts_on_positive,  # Isolate the contacts after individual test positive
        test_contacts_on_positive,  # Do we test contacts of a positive case immediately, or wait for them to develop symptoms
        do_symptom_testing,  # Test symptomatic individuals
        app_cov,  # Probability of tracing contact through app
        testing_delay,  # Delay between test and results
        do_manual_tracing,  # Perform manual tracing of contacts
        do_app_tracing,  # Perform app tracing of contacts
        app_trace_delay,  # Delay associated with tracing through the app
        manual_trace_delay,  # Delay associated with tracing manually
        manual_home_trace_prob,  # Probability of manually tracing a home contact
        manual_work_trace_prob,  # Probability of manually tracing a work contact
        manual_othr_trace_prob,  # Probability of manually tracing an other contact
        met_before_w,  # Probability of having met a work contact before to be able to manually trace
        met_before_s,  # Probability of having met a school contact before to be able to manually trace
        met_before_o,  # Probability of having met a other contact before to be able to manually trace
        max_contacts,  # Place a limit on the number of other contacts per day
        fractional_infections,  # Include infected but traced individuals as a fraction of their infection period not isolated
        quarantine_length,  # Length of quarantine imposed on COVID cases (and household)
        compliance,
        latent_period,  # Length of a cases incubation period (from infection to start of infectious period)
    ):
        self.rng = rng

        self.isolate_individual_on_symptoms = isolate_individual_on_symptoms
        self.isolate_individual_on_positive = isolate_individual_on_positive

        self.isolate_household_on_symptoms = isolate_household_on_symptoms
        self.isolate_household_on_positive = isolate_household_on_positive

        self.isolate_contacts_on_symptoms = isolate_contacts_on_symptoms
        self.isolate_contacts_on_positive = isolate_contacts_on_positive

        self.test_contacts_on_positive = test_contacts_on_positive

        self.do_symptom_testing = do_symptom_testing
        self.app_cov = app_cov

        self.testing_delay = testing_delay

        self.do_manual_tracing = do_manual_tracing
        self.do_app_tracing = do_app_tracing

        self.app_trace_delay = app_trace_delay
        self.manual_trace_delay = manual_trace_delay

        self.manual_home_trace_prob = manual_home_trace_prob
        self.manual_work_trace_prob = manual_work_trace_prob
        self.manual_othr_trace_prob = manual_othr_trace_prob

        self.met_before_w = met_before_w
        self.met_before_s = met_before_s
        self.met_before_o = met_before_o

        self.max_contacts = max_contacts

        self.fractional_infections = fractional_infections

        self.quarantine_length = quarantine_length
        self.compliance = compliance

        self.latent_period = latent_period

    def _trace_contacts(
        self, case, do_tracing, trace_prob, n_contacts, contacts_prevented
    ):
        contacts_traced = np.zeros(shape=n_contacts, dtype=bool)
        if do_tracing:
            if self.isolate_contacts_on_symptoms or (
                self.isolate_contacts_on_positive and case.covid
            ):
                contacts_traced = self.rng.binomial(
                    n=1, p=trace_prob, size=n_contacts
                ).astype(bool)

        contacts_traced = contacts_traced & ~contacts_prevented

        return contacts_traced

    def _isolate_contacts(self, n_contacts, contacts_traced, contacts_prevented):
        # Work out if each contact will adhere to the policy
        contacts_adherence = self.rng.binomial(
            n=1, p=self.compliance, size=n_contacts
        ).astype(bool)
        # Compute which contact will isolate because of the contact trace
        contacts_isolated = contacts_traced & contacts_adherence & ~contacts_prevented

        return contacts_isolated

    def _count_symptomatic_asymptomatic(
        self, n_contacts, contacts_infected, contacts_isolated
    ):
        contacts_symptomatic = (
            self.rng.binomial(
                n=1, p=config.PROP_COVID_SYMPTOMATIC, size=n_contacts
            ).astype(bool)
            & contacts_infected
        )
        symptomatic = contacts_isolated & contacts_infected & contacts_symptomatic
        asymptomatic = contacts_isolated & contacts_infected & ~contacts_symptomatic

        return symptomatic, asymptomatic

    def _count_contacts_quarantine_days(
        self,
        case,
        isolate_on_symptoms,
        isolate_on_positive,
        contacts_isolated,
        test_contacts_on_positive,
        contacts_tested_positive,
    ):
        # NOTE: working with "quarantined" as this represents those traced who were still contacted and complied
        if isolate_on_symptoms and not case.covid:
            return self.testing_delay * contacts_isolated.sum()
        elif (isolate_on_symptoms or isolate_on_positive) and case.covid:
            # NOTE: for now assume that people are tested on the same day as isolated.

            # If we are testing contacts on positive, then we will only need to quarantine those who are positive after the test
            # TODO: testing might be inefficient during latent period, perhaps we should quarantine contacts for latent_period and then test?
            if test_contacts_on_positive:
                # Those testing negative will spend testing_delay days in quarantine
                test_isolation_days = (
                    self.testing_delay
                    * (contacts_isolated & ~contacts_tested_positive).sum()
                )
                # Those who test positive will go into full quarantine
                quarantine_days = (
                    self.quarantine_length
                    * (contacts_isolated & contacts_tested_positive).sum()
                )

                return test_isolation_days + quarantine_days

            # Full quarantine for all contacts if not testing them
            return self.quarantine_length * contacts_isolated.sum()
        else:
            return 0

    def _get_contact_trace_delay(
        self,
        contacts_trace_app,
        contact_infections,
        contacts_isolated,
        contact_infection_days,
    ):
        # Trace delay is at max the manual trace delay
        trace_delay = self.manual_trace_delay * np.ones_like(contact_infection_days)

        # Contacts found via the app are traced with app_delay - assumed to be faster than manual
        contacts_trace_app_isolated = contacts_trace_app[
            contact_infections & contacts_isolated
        ]
        trace_delay[contacts_trace_app_isolated] = self.app_trace_delay

        return trace_delay

    def _get_fractional_metrics(
        self,
        case,
        test_perform_day,
        test_results_day,
        infection_days,
        trace_delay,
        isolate_on_symptoms,
        isolate_on_positive,
    ):
        infectiousness_by_day = case.inf_profile
        cumulative_infectiousness = np.cumsum(infectiousness_by_day)
        infectious_period = len(cumulative_infectiousness)

        # Compute day of contact becoming infectious after case started being infectious
        infectious_start = infection_days + self.latent_period

        if isolate_on_symptoms:
            days_not_quarantined = (test_perform_day + trace_delay) - infectious_start
        elif isolate_on_positive:
            days_not_quarantined = (test_results_day + trace_delay) - infectious_start
        else:
            # If neither of these are true, then the case would not have made it to here
            days_not_quarantined = infectious_period * np.ones(
                len(infectious_start), dtype=int
            )

        # clip the infectious period to max out at 1
        days_not_quarantined[
            days_not_quarantined > infectious_period
        ] = infectious_period

        fractional_num = len(days_not_quarantined)

        # Only care about ones where there is more than zero days spent unisolated
        days_not_quarantined = days_not_quarantined[days_not_quarantined > 0]

        # Add one to get indexing correct - 1st day infectious is 0 in array
        contact_cumulative_infectiousness = cumulative_infectiousness[
            days_not_quarantined - 1
        ].sum()

        return fractional_num, contact_cumulative_infectiousness

    def tti_chronology(self, case, case_factors):
        """Calculate if and when a case was tested and isolated

        Args:
            case:
            case_factors:

        Returns:
        """
        test_performed = case_factors.report_app or case_factors.report_manual
        if not test_performed:
            return test_performed, float("inf"), float("inf"), float("inf")
        else:
            test_perform_day = case.day_noticed_symptoms
            test_results_day = test_perform_day + self.testing_delay

            if self.isolate_individual_on_symptoms:
                isolate_day = test_perform_day
            elif self.isolate_individual_on_positive and case.covid:
                isolate_day = test_results_day
            else:
                # Don't isolate, set to something beyond simulation horizon
                # Float is fine here as long as used only for comparison
                isolate_day = float("inf")
            return test_performed, test_perform_day, test_results_day, isolate_day

    def __call__(self, case, contacts, case_factors):  # noqa: C901
        """Run a TTI strategy

        Args:
            case:
            contacts:
            case_factors:

        Returns:
        """
        (
            test_performed,
            test_perform_day,
            test_results_day,
            isolate_day,
        ) = self.tti_chronology(case, case_factors)

        # Days on which individual made contact with their contacts. For home, earliest day of infectivity.
        # home_contacts = contacts.home[:, 1]
        work_contacts = contacts.work[:, 1]
        othr_contacts = contacts.other[:, 1]

        # Get the day on which a household member was infected
        home_infected_day = contacts.home[:, 0]
        # Get if an infection was caused in contacts
        home_infections = (contacts.home[:, 0] >= 0).astype(bool)
        work_infections = (contacts.work[:, 0] >= 0).astype(bool)
        othr_infections = (contacts.other[:, 0] >= 0).astype(bool)

        n_home = home_infections.shape[0]
        n_work = work_infections.shape[0]
        n_othr = othr_infections.shape[0]

        # Compute reduction in contacts due to contact limiting policy. Independent of test status.
        othr_contacts_limited = ~_limit_contact(othr_contacts, self.max_contacts)

        # Compute reduction in contacts due to wfh. Independent of test status.
        if case_factors.wfh:
            work_contacts_wfh_limited = np.ones_like(work_contacts).astype(bool)
        else:
            work_contacts_wfh_limited = np.zeros_like(work_contacts).astype(bool)

        if test_performed:
            # Prevent contact after isolation day
            home_contacts_prevented = (home_infected_day >= isolate_day).astype(bool)
            work_contacts_prevented = (work_contacts >= isolate_day).astype(bool)
            othr_contacts_prevented = (othr_contacts >= isolate_day).astype(bool)

            # Remove contacts not made due to work from home
            work_contacts_prevented = (
                work_contacts_prevented | work_contacts_wfh_limited
            )

            # Remove other contact limiting contacts
            othr_contacts_prevented = othr_contacts_prevented | othr_contacts_limited

            # TRACING CONTACTS
            work_contacts_trace_app = self._trace_contacts(
                case,
                self.do_app_tracing and case_factors.report_app,
                self.app_cov,
                n_work,
                work_contacts_prevented,
            )
            othr_contacts_trace_app = self._trace_contacts(
                case,
                self.do_app_tracing and case_factors.report_app,
                self.app_cov,
                n_othr,
                othr_contacts_prevented,
            )
            # Even if the primary case reported symptoms via the app, we do manual tracing anyway as a safety net
            work_contacts_trace_manual = self._trace_contacts(
                case,
                self.do_manual_tracing,
                self.manual_work_trace_prob
                * (self.met_before_s if case.under18 else self.met_before_w),
                n_work,
                work_contacts_prevented,
            )
            othr_contacts_trace_manual = self._trace_contacts(
                case,
                self.do_manual_tracing,
                self.manual_othr_trace_prob * self.met_before_o,
                n_othr,
                othr_contacts_prevented,
            )

            # Assume all home contacts traced
            if self.isolate_household_on_symptoms or (
                self.isolate_household_on_positive and case.covid
            ):
                home_contacts_traced = np.ones_like(n_home, dtype=bool)
            else:
                home_contacts_traced = np.zeros(shape=n_home, dtype=bool)

            # Traced if traced either way and didn't isolate and prevent contact
            work_contacts_traced = work_contacts_trace_app | work_contacts_trace_manual
            othr_contacts_traced = othr_contacts_trace_app | othr_contacts_trace_manual

            # Compute trace statistics
            # Only trace if we want to isolate
            if self.isolate_contacts_on_symptoms or (
                self.isolate_contacts_on_positive and case.covid
            ):
                manual_traces = (
                    work_contacts_trace_manual.sum() + othr_contacts_trace_manual.sum()
                )
                app_traces = (
                    work_contacts_trace_app.sum() + othr_contacts_trace_app.sum()
                )
            else:
                manual_traces = 0.0
                app_traces = 0.0

            home_contacts_isolated = self._isolate_contacts(
                n_home, home_contacts_traced, home_contacts_prevented
            )
            work_contacts_isolated = self._isolate_contacts(
                n_work, work_contacts_traced, work_contacts_prevented
            )
            othr_contacts_isolated = self._isolate_contacts(
                n_othr, othr_contacts_traced, othr_contacts_prevented
            )

            # Do tests on the positive contacts if we want to, and find out which are asymptomatic
            if self.test_contacts_on_positive:
                (
                    work_tested_symptomatic,
                    work_tested_asymptomatic,
                ) = self._count_symptomatic_asymptomatic(
                    n_work, work_infections, work_contacts_isolated
                )
                work_tested_positive = (
                    work_tested_symptomatic & work_tested_asymptomatic
                )

                (
                    othr_tested_symptomatic,
                    othr_tested_asymptomatic,
                ) = self._count_symptomatic_asymptomatic(
                    n_othr, othr_infections, othr_contacts_isolated
                )
                othr_tested_positive = (
                    othr_tested_symptomatic & othr_tested_asymptomatic
                )
            else:
                work_tested_positive = None
                othr_tested_positive = None

            total_tests_performed = 0
            # count own test
            # TODO: Janky - if no contact tracing is going on, do NOT test the person
            if any(
                [
                    self.do_app_tracing,
                    self.do_manual_tracing,
                    self.isolate_contacts_on_positive,
                    self.isolate_contacts_on_symptoms,
                ]
            ):
                total_tests_performed += 1

            # If house isolated on symptoms, or on positive
            # These tests will not count against the primary case, as these would have been tested regardless.
            if case.covid:
                total_tests_performed += 0.0  # home_contacts_isolated.sum()

            # If contacts isolated on symptoms, or on positive
            # TODO: Again, after conversations, we will not test traced contacts unless a particular policy decision is made.
            # We do not count cases that would become positive and symptomatic against the primary case, but do count others.
            if case.covid:  # and test_contacts_on_positive:
                total_tests_performed += (
                    0  # work_contacts_isolated.sum() + othr_contacts_isolated.sum()
                )

            # Test contacts on positive test of the primary case. Only count the test excluding the symptomatic cases
            if self.test_contacts_on_positive and case.covid:
                total_tests_performed += (
                    work_contacts_isolated & ~work_tested_symptomatic
                ).sum()
                total_tests_performed += (
                    othr_contacts_isolated & ~othr_tested_symptomatic
                ).sum()

            # Compute the quarantine days
            person_days_quarantine = 0

            # If person has covid, require full quarantine
            if case.covid and (
                self.isolate_individual_on_symptoms
                or self.isolate_individual_on_positive
            ):
                person_days_quarantine += self.quarantine_length
            # If not, only require the test delay days of quarantine
            elif self.isolate_individual_on_symptoms:
                person_days_quarantine += self.testing_delay
            # Don't add any if: not isolating at all, individual only isolating after test complete

            person_days_quarantine += self._count_contacts_quarantine_days(
                case,
                self.isolate_household_on_symptoms,
                self.isolate_household_on_positive,
                home_contacts_isolated,
                # irrelevant parameters for household
                test_contacts_on_positive=False,
                contacts_tested_positive=None,
            )
            person_days_quarantine += self._count_contacts_quarantine_days(
                case,
                self.isolate_contacts_on_symptoms,
                self.isolate_contacts_on_positive,
                work_contacts_isolated,
                self.test_contacts_on_positive,
                work_tested_positive,
            )
            person_days_quarantine += self._count_contacts_quarantine_days(
                case,
                self.isolate_contacts_on_symptoms,
                self.isolate_contacts_on_positive,
                othr_contacts_isolated,
                self.test_contacts_on_positive,
                othr_tested_positive,
            )

        else:
            # No tracing took place if they didn't get tested positive.
            home_contacts_isolated = np.zeros(shape=n_home, dtype=bool)
            work_contacts_isolated = np.zeros(shape=n_work, dtype=bool)
            othr_contacts_isolated = np.zeros(shape=n_othr, dtype=bool)

            # Default cases prevented (none)
            home_contacts_prevented = np.zeros(shape=n_home, dtype=bool)
            work_contacts_prevented = work_contacts_wfh_limited
            othr_contacts_prevented = othr_contacts_limited

            manual_traces = 0
            app_traces = 0
            total_tests_performed = 0

            person_days_quarantine = 0

        # Compute the base reproduction rate
        base_rr = home_infections.sum() + work_infections.sum() + othr_infections.sum()

        # Compute the reproduction rate due to the policy
        # Remove infections due to case isolation
        home_infections_post_isolation = home_infections & ~home_contacts_prevented
        work_infections_post_isolation = work_infections & ~work_contacts_prevented
        othr_infections_post_isolation = othr_infections & ~othr_contacts_prevented

        # Count traced contacts as not included in the R TODO: make a proportion
        home_infections_post_policy = (
            home_infections_post_isolation & ~home_contacts_isolated
        )
        work_infections_post_policy = (
            work_infections_post_isolation & ~work_contacts_isolated
        )
        othr_infections_post_policy = (
            othr_infections_post_isolation & ~othr_contacts_isolated
        )

        # Count fractional cases - will only occur if got tested
        if test_performed and self.fractional_infections:

            # Get the days on which infections that were quarantined happened
            home_infection_days = home_infected_day[
                home_infections & home_contacts_isolated
            ]
            work_infection_days = work_contacts[
                work_infections & work_contacts_isolated
            ]
            othr_infection_days = othr_contacts[
                othr_infections & othr_contacts_isolated
            ]

            work_trace_delay = self._get_contact_trace_delay(
                work_contacts_trace_app,
                work_infections,
                work_contacts_isolated,
                work_infection_days,
            )
            othr_trace_delay = self._get_contact_trace_delay(
                othr_contacts_trace_app,
                othr_infections,
                othr_contacts_isolated,
                othr_infection_days,
            )
            # Home contacts traced immediately
            home_trace_delay = np.zeros_like(home_infection_days)

            (
                fractional_num_home,
                home_cumulative_infectiousness,
            ) = self._get_fractional_metrics(
                case,
                test_perform_day,
                test_results_day,
                home_infection_days,
                home_trace_delay,
                self.isolate_household_on_symptoms,
                self.isolate_household_on_positive,
            )
            (
                fractional_num_work,
                work_cumulative_infectiousness,
            ) = self._get_fractional_metrics(
                case,
                test_perform_day,
                test_results_day,
                work_infection_days,
                work_trace_delay,
                self.isolate_contacts_on_symptoms,
                self.isolate_contacts_on_positive,
            )
            (
                fractional_num_othr,
                othr_cumulative_infectiousness,
            ) = self._get_fractional_metrics(
                case,
                test_perform_day,
                test_results_day,
                othr_infection_days,
                othr_trace_delay,
                self.isolate_contacts_on_symptoms,
                self.isolate_contacts_on_positive,
            )

            # fractional_num = (
            #     fractional_num_home + fractional_num_work + fractional_num_othr
            # )
            fractional_R = (
                home_cumulative_infectiousness
                + work_cumulative_infectiousness
                + othr_cumulative_infectiousness
            )
            # inverse_fractional_R = fractional_num - fractional_R
            # home_fractional_R = home_cumulative_infectiousness
            home_inverse_fractional_R = (
                fractional_num_home - home_cumulative_infectiousness
            )
        else:
            fractional_R = 0.0
            home_cumulative_infectiousness = 0.0
            # inverse_fractional_R = 0.0
            # home_fractional_R = 0.0
            home_inverse_fractional_R = 0.0

        # Count the reduced infection rate
        reduced_rr = (
            home_infections_post_policy.sum()
            + work_infections_post_policy.sum()
            + othr_infections_post_policy.sum()
            + fractional_R
        )

        social_distancing_infections_prevented = (
            work_contacts_wfh_limited & work_infections
        ).sum() + (othr_contacts_limited & othr_infections).sum()
        symptom_isolation_infections_prevented = (
            (home_contacts_prevented & home_infections).sum()
            + (work_contacts_prevented & work_infections).sum()
            + (othr_contacts_prevented & othr_infections).sum()
            + home_inverse_fractional_R
            - social_distancing_infections_prevented
        )
        contact_tracing_infections_prevented = (
            base_rr
            - reduced_rr
            - social_distancing_infections_prevented
            - symptom_isolation_infections_prevented
        )

        return {
            RETURN_KEYS.base_r: base_rr if case.covid else np.nan,
            RETURN_KEYS.reduced_r: reduced_rr if case.covid else np.nan,
            RETURN_KEYS.man_trace: manual_traces,
            RETURN_KEYS.app_trace: app_traces,
            RETURN_KEYS.tests: total_tests_performed,
            RETURN_KEYS.quarantine: person_days_quarantine,
            RETURN_KEYS.covid: case.covid,
            RETURN_KEYS.symptomatic: case.symptomatic,
            RETURN_KEYS.tested: test_performed and self.do_symptom_testing,
            RETURN_KEYS.secondary_infections: home_infections.sum()
            + work_infections.sum()
            + othr_infections.sum(),
            RETURN_KEYS.cases_prevented_social_distancing: social_distancing_infections_prevented,
            RETURN_KEYS.cases_prevented_symptom_isolating: symptom_isolation_infections_prevented,
            RETURN_KEYS.cases_prevented_contact_tracing: contact_tracing_infections_prevented,
            RETURN_KEYS.fractional_r: fractional_R - home_cumulative_infectiousness,
        }
