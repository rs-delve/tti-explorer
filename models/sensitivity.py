from itertools import product

from utils import Registry

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
        yield {CONFIG_KEY: dict(cfg, **dict(zip(ablations.keys(), comb)))}


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


if __name__ == "__main__":
    from config import Ablation

    cfg = dict(a=0, b=1, c=2)
    ablations = dict(b=Ablation(bounds=None, values=range(3)), c=Ablation(bounds=None, values=range(3)))
    
    print("grid ablation")
    for dct in grid_ablation(cfg, ablations):
        print(dct)
    
    print("axis ablation")
    for dct in axis_ablation(cfg, ablations):
        print(dct)

