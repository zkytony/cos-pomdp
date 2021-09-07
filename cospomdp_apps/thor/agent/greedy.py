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
import pomdp_py
from ..common import TOS_Action, ThorAgent
from .cospomdp_basic import ThorObjectSearchCosAgent, GridMapSearchRegion
from cospomdp.models.agent import build_cos_observation_model

class GreedyNbvAgent:
    """Greedy next-best-view agent.
    The agent maintains a set of weighted particles for
    each object. Select view point based on expected entropy.
    (this is a different strategy, but a similar vein)
    about the joint belief space of all objects.

    The implementation here assumes operating in grid map coordinates.

    The belief update is

    B'(s') = B'(s1', s2', ..., sn', starget')
           = Pr(s1', s2', ..., sn', starget' | B, a, z)n
           = Pr(s1', s2', ..., sn' | starget', B, a, z) Pr(starget' | B, a, z)

    Pr(s1', s2', ..., sn' | starget', B, a, z)
           = prod_i^n Pr(si' | starget', B, a, z)
           = prod_i^n (Pr(z | B, a, si', starget') Pr(si' | starget', B, a)) / Pr(z | B, a)
           = prod_i^n (Pr(z | a, si', starget') sum_{si,starget} Pr(si, starget, si' | starget', B, a)) / Pr(z | B, a)
           = prod_i^n (Pr(z | a, si', starget') sum_{si,starget} Pr(si' | si, starget, starget', B, a) Pr(si, starget | starget', B, a)) / Pr(z | B, a)
           = prod_i^n (Pr(z | a, si', starget') sum_{si,starget} Pr(si' | si) Pr(si, starget | starget', B)) / Pr(z | B, a)
           = prod_i^n (Pr(z | a, si', starget') sum_{si,starget} Pr(si' | si) Pr(si | starget', B) Pr(starget | starget', B)) / Pr(z | B, a)
           = prod_i^n (Pr(z | a, si', starget') sum_{starget} Pr(si' | starget', B) Pr(starget | starget', B)) / Pr(z | B, a)
           = prod_i^n (Pr(z | a, si', starget') Pr(si' | starget', B)) / Pr(z | B, a)

    Because we are given only conditionals on the target state, we do not have the information
    to compute Pr(z | a, si', starget') and Pr(si' | starget', B). Instead, we will approximate,
    by assuming that Pr(starget' | B, a, z) has incorporated the information about observation z,
    and the belief about an object i is updated after the belief about target is updated, according to:

    Pr(si' | B, a, z) = Pr(si' | starget', B, a, z) Pr(starget' | B, a, z)
                      = Pr(si' | starget') Pr(starget' | B, a, z)
    """
    def __init__(self, robot_id, target, search_region, corr_dists, detectors,
                 num_particles=100, prior={}):
        """
        prior: Maps from object_id to a map from search region location to a float.
        detectors: Maps from objid to a DetectionModel Pr(zi | si, srobot')
            Must contain an entry for the target object
        corr_dists: Maps from objid to CorrelationDist Pr(si | starget)
            Does not need to contain an entry for the target object
        """
        self.search_region = search_region
        self.robot_id = robot_id
        self.target = target
        self.particle_beliefs = {}
        # initialize particle beliefs.
        for objid in detectors:
            self.particle_beliefs[objid] = self._init_belief(objid, num_particles, prior=prior.get(objid, {}))

        # Constructs the CosObservationModel
        self.observation_model = build_cos_observation_model(
            corr_dists, detectors, robot_id, target[0])

    @property
    def detectable_objects(self):
        return list(self.detectors.keys())

    def _init_belief(self, objid, num_particles, prior={}):
        """prior: Maps from search region location to a float."""
        # For every detectable object, maintain a set of particle beliefs
        # The initial belief is
        if len(prior) > 0:
            hist = pomdp_py.Histogram(prior)
        else:
            hist = pomdp_py.Histogram({loc:1.0} for loc in self.search_region)
        particles = []
        for objid in self.detectable_objects:
            objclass = self.detectable_objects[objid][1]
            for i in range(num_particles):
                loc = hist.random()
                weight = hist[loc]
                si = self.search_region.object_state(
                    objid, objclass, loc
                )
                particles.append((si, weight))
        return particles

class ThorObjectSearchGreedyNbvAgent(ThorAgent):
    """Uses a GreedyNbvAgent to search in thor."""
    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 grid_map,
                 num_particles=100):
        """
        thor_prior: dict mapping from thor location to probability; If empty, then the prior will be uniform.
        """
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

        self.greedy_agent = GreedyNbvAgent(robot_id, target, search_region,
                                           corr_dists, detectors,
                                           num_particles=num_particles)

    def act(self):
        pass

    def update(self, tos_action, tos_observation):
        pass

    @property
    def detectable_objects(self):
        return self.greedy_agent.detectable_objects
