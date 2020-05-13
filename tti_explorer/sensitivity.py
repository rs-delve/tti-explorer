from itertools import product

from .utils import Registry

registry = Registry()

CONFIG_KEY = 'config'
TARGET_KEY = 'sensitivity_target'


@registry('grid')
def grid_ablation(cfg, ablations):
    """grid_ablation
    Try all combinations of values in ablations

    Args:
        cfg:
        ablations:

    Returns:
    """
    vals = (a.values for a in ablations.values() if a.values is not None)
    for comb in product(*vals):
        yield {CONFIG_KEY: dict(cfg, **dict(zip(ablations.keys(), comb))), TARGET_KEY: list(ablations.keys())}


@registry('axis')
def axis_ablation(cfg, ablations):
    """axis_ablation
    Vary one parameter at a time in ablations, keeping others fixed

    Args:
        cfg:
        ablations:

    Returns:
    """
    for k, ablation in ablations.items():
        for value in ablation.values:
            yield {CONFIG_KEY: dict(cfg, **{k: value}), TARGET_KEY: k}

