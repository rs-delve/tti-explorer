from itertools import product

from utils import Registry

registry = Registry()


@registry('grid')
def grid_ablation(cfg, ablations):
    vals = (a.values for a in ablations.values() if a.values is not None)
    for comb in product(*vals):
        yield dict(cfg, **dict(zip(ablations.keys(), comb)))



if __name__ == "__main__":
    from config import Ablation

    cfg = dict(a=0, b=1)
    ablations = dict(b=Ablation(bounds=None, values=range(10)))
    
    for dct in grid_ablation(cfg, ablations):
        print(dct)

