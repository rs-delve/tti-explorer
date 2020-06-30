from itertools import product

from . import config
from .utils import Registry

registry = Registry()

CONFIG_KEY = "config"
TARGET_KEY = "sensitivity_target"


@registry("grid")
def grid_variation(cfg, sensitivities):
    """grid_variation
    Try all combinations of values in variations

    Args:
        cfg (dict): Default configurations around which to vary
        sensitivities (dict[str: Sensitivity]): Values to vary through, expressed as
        Sensitivitys.

    Returns:
        configs (dict): Configuration dict, with appropriate parameters varied.
        This dictionary has two entries, one for the config and one for the
        parameter(s) being varied from the defaults in said config.
    """
    vals = (a.values for a in sensitivities.values() if a.values is not None)
    for comb in product(*vals):
        yield {
            CONFIG_KEY: dict(cfg, **dict(zip(sensitivities.keys(), comb))),
            TARGET_KEY: list(sensitivities.keys()),
        }


@registry("axis")
def axis_variation(cfg, sensitivities):
    """axis_variation
    Vary one parameter at a time in variations, keeping others fixed

    Args:
        cfg (dict): Default configurations around which to vary
        sensitivities (dict[str: Sensitivity]): Values to vary through, expressed as
        Sensitivitys.

    Returns:
        configs (dict): Configuration dict, with appropriate parameters varied.
        This dictionary has two entries, one for the config and one for the
        parameter(s) being varied from the defaults in said config.
    """
    for k, sensitivity in sensitivities.items():
        for value in sensitivity.values:
            yield {CONFIG_KEY: dict(cfg, **{k: value}), TARGET_KEY: k}


class SensitivityConfigGenerator():
    """
    Class for generating sensitivity configurations
    """

    def __init__(self, sensitivity_method, sensitivity_targets):
        """
        :param sensitivity_method: sensitivity method to generate configs for
        :param sensitivity_targets: parameters to vary in sensitivity analysis
        """
        if sensitivity_method is not None:
            self.generator = registry[sensitivity_method]
        else:
            self.generator = None
        self.sensitivity_targets = sensitivity_targets

    def generate_for_strategy(self, strategy_name, strategy_config):
        """
        Generates configs for a given strategy

        :param strategy_name: Name of a strategy
        :param strategy_config: Dictionary of strategy parameter settings
        :returns: List of sensitivity configurations.
                  Each config specifies strategy parameter values for a scenario run, and a current sensitivity target parameter.
        """
        policy_sensitivities = config.get_strategy_sensitivities(strategy_name)
        if self.sensitivity_targets:
            policy_sensitivities = dict((k, policy_sensitivities[k]) for k in self.sensitivity_targets)

        if self.generator:
            sensitivity_configs = self.generator(strategy_config, policy_sensitivities)
        else:
            sensitivity_configs = [{CONFIG_KEY: strategy_config, TARGET_KEY: ""}]

        return sensitivity_configs
