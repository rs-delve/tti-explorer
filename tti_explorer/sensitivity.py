from itertools import product

from .utils import Registry

registry = Registry()

CONFIG_KEY = 'config'
TARGET_KEY = 'sensitivity_target'


@registry('grid')
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
                TARGET_KEY: list(sensitivities.keys())
                }


@registry('axis')
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

