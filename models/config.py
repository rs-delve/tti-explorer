

class CaseConfig:
    p_symptomatic_covid_neg = 200 / 260
    p_symptomatic_covid_pos = 30 / 260
    p_asymptomatic_covid_pos = 30 / 260

    #Conditional on symptomatic
    p_has_app = 0.35
    # Conditional on having app
    p_report_app = 0.75
    p_report_nhs_g_app = 0.5

    # Conditional on not having app
    p_report_nhs_g_no_app = 0.5

    groups = {
            'symptomatic_covid': [
                p_symptomatic_covid_neg,
                p_symptomatic_covid_pos,
                p_asymptomatic_covid_pos
            ],
        }

    def __init__(self):
        for name, group in self.groups.items():
            if sum(group) != 1.:
                raise ValueError(
                        f"Probabilities in group {name} don't sum to 1. \n {group}\n"
                        )
