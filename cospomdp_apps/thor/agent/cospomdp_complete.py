# This agent implements the complete
#
#   POMDP <-> (Robot <-> World)
#
# framework; Here, POMDP is the COS-POMDP;
# It tracks both robot state at a topological graph node and
# the ground robot pose; The topological graph is automatically
# built based on sampling.
from ..common import ThorAgent

class ThorObjectSearchCompleteCosAgent(ThorAgent):
    AGENT_USES_CONTROLLER=True
