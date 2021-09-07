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
import math
import pomdp_py
import random

from cospomdp.models.agent import build_cos_observation_model
from cospomdp.models.sensors import yaw_facing
from cospomdp.utils.math import euclidean_dist
import cospomdp

from ..common import TOS_Action, ThorAgent
from .cospomdp_basic import (ThorObjectSearchCosAgent,
                             GridMapSearchRegion,
                             ThorObjectSearchBasicCosAgent)
from .cospomdp_complete import _shortest_path
from .components.action import MoveViewpoint
from .components.goal_handlers import MacroMoveHandler, DoneHandler

class GreedyNbvAgent:
    """Greedy next-best-view agent.
    The agent maintains a set of weighted particles for
    each object. Select view point based on a "hybrid utility"
    (this may not be exactly the same, but it is their idea)

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
    def __init__(self, target, init_robot_state,
                 search_region, reachable_positions,
                 corr_dists, detectors, h_angles,
                 goal_distance, num_particles=100, prior={},
                 done_thres=0.9,
                 num_viewpoint_samples=10,
                 decision_params={}):
        """
        prior: Maps from object_id to a map from search region location to a float.
        detectors: Maps from objid to a DetectionModel Pr(zi | si, srobot')
            Must contain an entry for the target object
        corr_dists: Maps from objid to CorrelationDist Pr(si | starget)
            Does not need to contain an entry for the target object
        reachable_positions (list): List of 2D (x,y) locations that are
             possible locations for the robot to reach
        alpha, beta, sigma are parameters from the paper.
        """
        self.search_region = search_region
        self.reachable_positions = reachable_positions
        self.detectors = detectors
        self._init_robot_state = init_robot_state
        self.target = target
        self.brobot = pomdp_py.WeightedParticles([(self._init_robot_state, 1.0)])
        self.particle_beliefs = {}
        self._num_particles = num_particles
        self._done_thres = 0.9
        self._h_angles = h_angles
        self._num_viewpoint_samples = num_viewpoint_samples
        self._decision_params = decision_params
        self._goal_distance = goal_distance
        # initialize particle beliefs.
        for objid in detectors:
            self.particle_beliefs[objid] = self._init_belief(objid, prior=prior.get(objid, {}))

        # Constructs the CosObservationModel
        self.observation_model = build_cos_observation_model(
            corr_dists, detectors, self.robot_id, target[0])

        self._current_goal = None
        self.last_viewpoints = []

    def sensor(self, objid):
        return self.observation_model.zi_models[objid].detection_model.sensor

    @property
    def belief(self):
        return pomdp_py.OOBelief({**self.particle_beliefs,
                                  **{self.robot_id: self.brobot}})

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
        for cls in self.detectable_objects:
            for i in range(self._num_particles):
                if hist is not None:
                    loc = hist.random()
                    weight = hist[loc]
                else:
                    loc = random.sample(self.search_region.locations, 1)[0]
                    weight = 1.0 / len(self.search_region.locations)
                si = self.search_region.object_state(
                    objid, cls, loc
                )
                particles.append((si, weight))
        return pomdp_py.WeightedParticles(particles)

    def update(self, action, observation):
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
        next_btarget = self._update_target_particles(btarget, next_srobot, observation, self._num_particles)
        new_particle_beliefs = {self.target_id : next_btarget}
        for objid in self.particle_beliefs:
            if objid != self.target_id:
                bobj = self.particle_beliefs[objid]
                next_bobj = self._update_object_particles(objid, bobj, observation,
                                                          next_btarget, next_srobot,
                                                          self._num_particles)
                new_particle_beliefs[objid] = next_bobj
        self.particle_beliefs = new_particle_beliefs

    def _reinvigorate(self, particles):
        """
        Quote from paper:
        To deal with particle decay, we reinvigorate the particles of each o i by
        sampling in known room areas, as well as around other objects o j based on B(Ri
        j). In step 5, j ∈ Γ(i) only if 1−B(Ri j = Disjoint) > 0.2. Across our
        experiments, we use 100 particles for each object.

        Because we don't have B(Rj), we will only do the first kind of reinvigoration

        Args:
            particles (WeightedParticles)
        """
        particles = particles.condense()
        _srnd = particles.random()
        _objid = _srnd.id
        _cls = _srnd.objclass
        new_particles = [p for p in particles.particles]
        if len(particles) <= self._num_particles:
            for _ in range(self._num_particles - len(particles)):
                # sample in the search region
                loc = random.sample(self.search_region.locations, 1)[0]
                si = self.search_region.object_state(
                    _objid, _cls, loc
                )
                if si in particles:
                    weight = particles[si]
                else:
                    weight = 1.0 / len(self.search_region.locations)
                new_particles.append((si, weight))
        return pomdp_py.WeightedParticles(new_particles)

    def _update_target_particles(self, btarget, next_srobot, observation, num_particles):
        _temp_particles = []
        for _ in range(num_particles):
            starget = btarget.random()
            next_state = cospomdp.CosState({self.target_id: starget,
                                            next_srobot.id: next_srobot})
            weight = self.observation_model.probability(observation, next_state)
            _temp_particles.append((starget, weight))
        # resampling
        _temp_particles = pomdp_py.WeightedParticles(_temp_particles)
        resampled_particles = []
        for _ in range(num_particles):
            starget = _temp_particles.random()
            weight = _temp_particles[starget]
            resampled_particles.append((starget, weight))

        belief = pomdp_py.WeightedParticles(resampled_particles)
        return self._reinvigorate(belief)

    def _update_object_particles(self, objid, bobj, observation, next_btarget, next_srobot, num_particles):
        """Particle Filter that follows
        Pr(si' | B, a, z) = Pr(si' | starget', B, a, z) Pr(starget' | B, a, z)
                         ~= Pr(si' | starget', B, a, zi) Pr(starget' | B, a, z)
                          = Pr(zi | si', starget') Pr(si' | starget', B) Pr(starget' | B, a, z)
                          = Pr(zi | si') B(si') * Pr(si' | starget') Pr(starget' | B, a, z)

        Note that Pr(starget' | B, a, z) has been updated already."""
        _temp_particles = []
        for _ in range(num_particles):
            starget = next_btarget.random()
            dist_si = self.observation_model.z(objid).corr_cond_dist(starget)
            si = dist_si.sample()
            zi = observation.z(objid)
            zi_prob = self.observation_model.z(objid).detection_model.probability(zi, si, next_srobot)
            weight = zi_prob
            _temp_particles.append((si, weight))
        _temp_particles = pomdp_py.WeightedParticles(_temp_particles)
        resampled_particles = []
        for _ in range(num_particles):
            si = _temp_particles.random()
            weight = _temp_particles[starget]
            resampled_particles.append((si, weight))
        belief = pomdp_py.WeightedParticles(resampled_particles)
        return self._reinvigorate(belief)

    def act(self):
        """Quote from Zeng et al. 2020

        UHS encourages the robot to explore promising areas that could
        contain the target object and/or any landmark object, while accounting
        for navigation cost

        0. If the highest belief is above some threshold,
           and the robot is within goal distance facing the object, then Done.
        1. If currently not moving towards a viewpoint, then selects view points
        2. Decides which view point to visit based on a "hybrid utility function"
          - This will be a high-level goal; The A* planner is expected to handle
           this. Will not replan when the A* goal is not reached.
        """
        if self._current_goal is not None:
            return self._current_goal

        # sample view points based on belief over the target location
        srobot = self.brobot.mpe()
        btarget = self.particle_beliefs[self.target_id]
        starget_mpe = btarget.mpe()
        print("MPE belief:", btarget[starget_mpe])
        if btarget[starget_mpe] > self._done_thres:
            if euclidean_dist(starget_mpe.loc, srobot.loc) <= self._goal_distance\
               and self.sensor(self.target_id).in_range_facing(
                   starget_mpe.loc, srobot.pose):
                return cospomdp.Done()
            else:
                # set the view point to be location closest to the target.
                robot_pos = min(self.reachable_positions,
                                key=lambda p: euclidean_dist(p, starget_mpe.loc))
                yaw = yaw_facing(srobot.loc, starget_mpe.loc, self._h_angles)
                self._current_goal = MoveViewpoint((*robot_pos, yaw))
                return self._current_goal

        viewpoints = []
        for _ in range(self._num_viewpoint_samples):
            starget = btarget.random()
            # a view point is a pose. Get the position from starget,
            # and choose the robot reachable position closest to the target,
            # then get the yaw facing the target
            robot_pos = min(self.reachable_positions, key=lambda p: euclidean_dist(p, starget.loc))
            yaw = yaw_facing(srobot.loc, starget.loc, self._h_angles)
            robot_pose = (*robot_pos, yaw)
            viewpoints.append((robot_pose, starget, btarget[starget]))

        # decide which view point to visit
        best_score = float('-inf')
        best_viewpoint = None
        alpha = self._decision_params.get("alpha", 0.1) # numbers from the paper
        beta = self._decision_params.get("beta", 0.4)
        sigma = self._decision_params.get("sigma", 0.5)
        for robot_pose, starget, weight_target in viewpoints:
            # because path is over grid map, its length is the length of the path
            navigation_distance = len(_shortest_path(self.reachable_positions,
                                                     srobot.loc, robot_pose[:2]))
            # Trades off going for the target with any other object.
            # We sample other object states conditioned on the target object location.
            # and check if the robot can observe them from the view point
            weight_observing_other_object = 0
            for objid in self.particle_beliefs:
                if objid != self.target_id:
                    # sample other object locations conditioned on target location
                    zi_model = self.observation_model.zi_models[objid]
                    dist_si = zi_model.corr_cond_dist(starget)
                    si = dist_si.sample()
                    # check if the robot can observe this object
                    imagined_srobot = self._init_belief.__class__(self.robot_id, robot_pose)
                    zi = self.observation_model.z(objid).sample(si, imagined_srobot)
                    if zi.loc is not None:
                        identity = 1.0
                    else:
                        identity = 0.0
                    weight_si = dist_si[si]
                    weight_observing_other_object = max(weight_observing_other_object,
                                                        weight_si * identity)

            score = weight_target + alpha * 1 / (sigma * math.atan(navigation_distance))\
                + beta * weight_observing_other_object
            if score > best_score:
                best_viewpoint = robot_pose

        self._current_goal = MoveViewpoint(robot_pose, )
        self.last_viewpoints = ([pt[0] for pt in viewpoints], best_viewpoint)
        return self._current_goal

    def clear_goal(self):
        self._current_goal = None


class ThorObjectSearchGreedyNbvAgent(ThorObjectSearchCosAgent):
    """Uses a GreedyNbvAgent to search in thor."""
    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 grid_map,
                 thor_agent_pose,
                 num_particles=100,
                 done_thres=0.9,
                 num_viewpoint_samples=10,
                 decision_params={}):
        """
        thor_prior: dict mapping from thor location to probability; If empty, then the prior will be uniform.
        """
        robot_id = task_config['robot_id']
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.grid_map = grid_map
        self.search_region = search_region
        self.reachable_positions = reachable_positions

        init_robot_pose = grid_map.to_grid_pose(
            thor_agent_pose[0][0],  #x
            thor_agent_pose[0][2],  #z
            thor_agent_pose[1][1]   #yaw
        )

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
        self.thor_movement_params = task_config["nav_config"]["movement_params"]

        detectors, detectable_objects = ThorObjectSearchCosAgent.build_detectors(
            self.task_config["detectables"], detector_specs)
        corr_dists = ThorObjectSearchCosAgent.build_corr_dists(
            self.target[0], self.search_region, corr_specs, detectable_objects)

        init_robot_state = cospomdp.RobotState2D(robot_id, init_robot_pose)
        h_angles = self.task_config['nav_config']['h_angles']
        goal_distance = task_config["nav_config"]["goal_distance"] / grid_map.grid_size
        self.greedy_agent = GreedyNbvAgent(target, init_robot_state,
                                           search_region, reachable_positions,
                                           corr_dists, detectors, h_angles,
                                           goal_distance=goal_distance,
                                           num_particles=num_particles,
                                           done_thres=done_thres,
                                           num_viewpoint_samples=num_viewpoint_samples,
                                           decision_params=decision_params)
        self._goal_handler = None

    @property
    def cos_agent(self):
        """greedy agent is not a COS-POMDP agent,
        but it is indeed an agent for correlationl object search."""
        return self.greedy_agent

    def act(self):
        goal = self.greedy_agent.act()
        print("Goal: {}".format(goal))
        if isinstance(goal, cospomdp.Done):
            self._goal_handler = DoneHandler(goal, self)
            return self._goal_handler.step()

        if self._goal_handler is None\
           or goal != self._goal_handler.goal\
           or self._goal_handler.done:
            assert isinstance(goal, MoveViewpoint)
            self._goal_handler = MacroMoveHandler(goal.dst_pose[:2], self,
                                                  rot=(0, goal.dst_pose[2], 0),
                                                  angle_tolerance=5,
                                                  goal=goal)

            if self._goal_handler.done:
                # goal handler is done (immediately when it is created)
                self._goal_handler = DoneHandler(goal, self)
                return self._goal_handler.step()

        return self._goal_handler.step()

    def update(self, tos_action, tos_observation):
        if self._goal_handler.updates_first:
            self._goal_handler.update(tos_action, tos_observation)

        super().update(tos_action, tos_observation)

        if not self._goal_handler.updates_first:
            self._goal_handler.update(tos_action, tos_observation)

        if self._goal_handler.done:
            print("Goal reached.")
            self.greedy_agent.clear_goal()

    @property
    def detectable_objects(self):
        return self.greedy_agent.detectable_objects

    @property
    def belief(self):
        return self.greedy_agent.belief

    def _update_belief(self, action, observation):
        self.greedy_agent.update(action, observation)

    def interpret_robot_obz(self, tos_observation):
        return ThorObjectSearchBasicCosAgent.interpret_robot_obz(self, tos_observation)

    def interpret_action(self, tos_action):
        # It actually doesn't matter to the greedy what low-level
        # action is taken by the goal handler.
        return None
