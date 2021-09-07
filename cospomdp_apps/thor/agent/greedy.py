"""This agent is based on Zeng et al.
"Semantic Linking Maps for Active Visual Object Search"
Although parts are simplified.

Quote from the paper:
1. We propose an active visual object search strategy
method through our introduction of the Semantic Linking Maps
(SLiM) model. SLiM simultaneously maintains the belief over a
target object’s location as well as landmark objects’ locations,
while accounting for probabilistic inter-object spatial relations.
(We also maintain joint belief of particles and update based
on correlation - this should be the same idea; It is just we
do not have access to conditionals between other objects)

2. we describe a hybrid search strategy that selects the next best view pose for
searching for the target object based on the maintained belief.
(We do not do this but do something along a similar line_
"""
from ..common import TOS_Action, ThorAgent
from .cospomdp_basic import ThorObjectSearchCosAgent, GridMapSearchRegion

class ThorObjectSearchGreedyNbvAgent(ThorAgent):

    """The agent maintains a set of weighted particles for
    each object. Select view point based on expected entropy.
    (this is a different strategy, but a similar vein)
    about the joint belief space of all objects."""

    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 grid_map):
        robot_id = task_config['robot_id']
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.grid_map = grid_map
        self.search_region = search_region
        self.reachable_positions = reachable_positions

        if task_config["task_type"] == 'class':
            target_id = task_config['target']
            target_class = task_config['target']
            target = (target_id, target_class)
        else:
            # This situation is not tested :todo:
            target = task_config['target']  # (target_id, target_class)
            target_id = target[0]
        self.task_config = task_config
        self.target = target

        detectors, detectable_objects = ThorObjectSearchCosAgent.build_detectors(
            self.task_config["detectables"], detector_specs)
        corr_dists = ThorObjectSearchCosAgent.build_corr_dists(
            self.target[0], self.search_region, corr_specs, detectable_objects)
        self.detectable_objects = detectable_objects
        self.corr_dists = corr_dists



    def act(self):
        pass

    def update(self, tos_action, tos_observation):
        pass

    def _init_belief(self):
        pass
