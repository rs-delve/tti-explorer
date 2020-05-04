
from utils import Registry

registry = Registry()

# How to define a scenario
@registry('no_measures')
def function_that_does_no_measures(*args, **kwargs):
    return 10
