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
import random
from ..common import TOS_Action, ThorAgent
from .cospomdp_basic import ThorObjectSearchCosAgent, GridMapSearchRegion
from cospomdp.models.agent import build_cos_observation_model
from cospomdp.domain.state import CosState

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
    def __init__(self, target, init_robot_state, search_region, corr_dists, detectors,
                 num_particles=100, prior={}):
        """
        prior: Maps from object_id to a map from search region location to a float.
        detectors: Maps from objid to a DetectionModel Pr(zi | si, srobot')
            Must contain an entry for the target object
        corr_dists: Maps from objid to CorrelationDist Pr(si | starget)
            Does not need to contain an entry for the target object
        """
        self.search_region = search_region
        self._init_robot_state = init_robot_state
        self.target = target
        self.brobot = pomdp_py.WeightedParticles([(self._init_robot_state, 1.0)])
        self.particle_beliefs = {}
        self._num_particles = num_particles
        # initialize particle beliefs.
        for objid in detectors:
            self.particle_beliefs[objid] = self._init_belief(objid, prior=prior.get(objid, {}))

        # Constructs the CosObservationModel
        self.observation_model = build_cos_observation_model(
            corr_dists, detectors, self.robot_id, target[0])

    @property
    def robot_id(self):
        return self._init_robot_state.id

    @property
    def target_id(self):
        return self.target[0]

    @property
    def detectable_objects(self):
        return list(self.detectors.keys())

    def _init_belief(self, objid, prior={}):
        """prior: Maps from search region location to a float."""
        # For every detectable object, maintain a set of particle beliefs
        # The initial belief is
        if len(prior) > 0:
            hist = pomdp_py.Histogram(prior)
        else:
            hist = None
        particles = []
        for objid in self.detectable_objects:
            objclass = self.detectable_objects[objid][1]
            for i in range(self._num_particles):
                if hist is not None:
                    loc = hist.random()
                else:
                    loc = random.sample(self.search_region.locations, 1)[0]

                weight = hist[loc]
                si = self.search_region.object_state(
                    objid, objclass, loc
                )
                particles.append((si, weight))
        return pomdp_py.WeightedParticles(particles)

    def update_belief(self, observation):
        """
        Args:
            observation (CosObservation)
        """
        robotobz = observation.z(self.robot_id)
        rstate_class = self._init_robot_state.__class__
        next_srobot = rstate_class.from_obz(robotobz)
        next_brobot = pomdp_py.WeightedParticles([(next_srobot, 1.0)])
        self.brobot = next_brobot

        btarget = self.particle_beliefs[self.target_id]
        next_btarget = self._update_particle_belief(btarget, observation, is_target=True)
        new_particle_beliefs = {self.target_id : next_btarget}
        for objid in self.particle_beliefs:
            if objid != self.targte_id:
                bobj = self.particle_beliefs[objid]
                next_bobj = self._update_object_particles(objid, bobj, observation,
                                                          next_btarget, next_srobot,
                                                          self._num_particles)
                new_particle_beliefs[objid] = next_bobj
        self.particle_beliefs = new_particle_beliefs

    def _update_target_particles(self, btarget, next_srobot, observation, num_particles):
        _temp_particles = []
        for i in range(num_particles):
            starget = btarget.random()
            next_state = CosState({self.target_id: starget,
                                   next_srobot.id: next_srobot})
            weight = self.observation_model.probability(observation, next_state)
            _temp_particles.append((starget, weight))
        # resampling
        _temp_particles = pomdp_py.WeightedParticles(_temp_particles)
        resampled_particles = []
        for i in range(num_particles):
            starget = _temp_particles.random()
            weight = _temp_particles[starget]
            resampled_particles.append((starget, weight))
        return pomdp_py.WeightedParticles(resampled_particles)

    def _update_object_particles(self, objid, bobj, observation, next_btarget, next_srobot, num_particles):
        """Particle Filter that follows
        Pr(si' | B, a, z) = Pr(si' | starget', B, a, z) Pr(starget' | B, a, z)
                         ~= Pr(si' | starget', B, a, zi) Pr(starget' | B, a, z)
                          = Pr(zi | si', starget') Pr(si' | starget', B) Pr(starget' | B, a, z)
                          = Pr(zi | si') B(si') * Pr(si' | starget') Pr(starget' | B, a, z)

        Note that Pr(starget' | B, a, z) has been updated already."""
        _temp_particles = []
        for i in range(num_particles):
            starget = next_btarget.random()
            dist_si = self.observation_model.z(objid).corr_cond_dist(starget)
            si = dist_si.sample()
            zi = observation.z(objid)
            zi_prob = self.observation_model.z(objid).detection_model.probability(zi, si, next_srobot)
            weight = zi_prob
            _temp_particles.append((si, weight))
        _temp_particles = pomdp_py.WeightedParticles(_temp_particles)
        resampled_particles = []
        for i in range(num_particles):
            si = _temp_particles.random()
            weight = _temp_particles[starget]
            resampled_particles.append((si, weight))
        return pomdp_py.WeightedParticles(resampled_particles)


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
