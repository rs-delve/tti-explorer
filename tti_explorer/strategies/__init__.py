from ..utils import Registry

from .temporal_anne_flowchart import temporal_anne_flowchart
from .cmmid import CMMID_strategy
from .cmmid_better import CMMID_strategy_better


registry = Registry()
temporal_anne_flowchart = registry('temporal_anne_flowchart')(temporal_anne_flowchart)
cmmid = registry('CMMID')(cmmid)
cmmid_better = registry('CMMID_better')(cmmid_better)


from .common import RETURN_KEYS
