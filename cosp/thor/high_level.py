########################### COS-POMDP agent #################################
import math
import random
import pomdp_py
from pomdp_py import Agent as POMDP
from thortils import thor_object_type
from . import constants
from .decisions import MoveDecision, SearchDecision, DoneDecision
from ..models import cospomdp
from ..utils.math import indicator, normalize, euclidean_dist
from ..utils.misc import resolve_robot_target_args

class HighLevelStatus:
    MOVING = "moving"
    SEARCHING = "searching"
    INITIAL = "initial"
    DONE = "done"

class HighLevelRobotState(pomdp_py.ObjectState):
    def __init__(self, pos, status):
        super().__init__("robot", dict(pos=pos, status=status))

class HighLevelObjectState(pomdp_py.ObjectState):
    """The `pos` means the location that the robot should be in
    if it wants to detect the object."""
    def __init__(self, object_class, pos):
        super().__init__(object_class, dict(pos=pos))

class HighLevelOOBelief(pomdp_py.OOBelief):
    def __init__(self, robot_id, target_id, *args):
        self.robot_id = robot_id
        self.target_id = target_id
        robot_belief, target_belief =\
            resolve_robot_target_args(robot_id,
                                      target_id,
                                      *args)
        super().__init__({self.robot_id: robot_belief,
                          self.target_id: target_belief})

    def random(self, rnd=random):
        return cospomdp.ReducedState(self.robot_id, self.target_id,
                                     super().random(rnd=random, return_oostate=False))
    def mpe(self, rnd=random):
        # import pdb; pdb.set_trace()
        return cospomdp.ReducedState(self.robot_id, self.target_id,
                                     super().mpe(return_oostate=False))
    @property
    def target_belief(self):
        return self.object_beliefs[self.target_id]

class HighLevelSearchRegion(cospomdp.SearchRegion):
    """This is where the high-level belief will be defined over.
    Instead of meaning 'the location of the target', it means
    'the location that the robot should be in if it wants to
    detect the target'. This means the robot and the target
    occupy on the same search region."""
    def __init__(self, reachable_positions):
        """reachable_positions: (initial) set of 2D locations the robot can reach."""
        self.reachable_positions = set(reachable_positions)
        self._current = 0

    def __iter__(self):
        return iter(self.reachable_positions)

    def __contains__(self, item):
        return item in self.reachable_positions

    def neighbor(self, pos):
        x, z = pos
        return filter(lambda p: p in self.reachable_positions,
                      [(x + constants.GRID_SIZE, z),
                       (x - constants.GRID_SIZE, z),
                       (x, z + constants.GRID_SIZE),
                       (x, z - constants.GRID_SIZE)])

class HighLevelTransitionModel(pomdp_py.TransitionModel):
    """Transition model for high-level planner.

    If the decision is moving to another position, then simply change the robot
    pose to that position (deterministic). If the decision is search, then
    update robot state status to be searching locally. If done, then done.
    """
    def __init__(self, robot_id, target_id):
        self.robot_id = robot_id
        self.target_id = target_id

    def sample(self, state, action):
        """
        Args:
            state (cospomdp.ReducedState)
            action (Decision)
        """
        next_robot_pos = state.robot_state["pos"]
        next_robot_status = state.robot_state["status"]
        if isinstance(action, MoveDecision):
            next_robot_pos = action.dest
            next_robot_status = HighLevelStatus.MOVING
        elif isinstance(action, SearchDecision):
            next_robot_status = HighLevelStatus.SEARCHING
        next_robot_state = HighLevelRobotState(next_robot_pos,
                                               next_robot_status)
        target_state = HighLevelObjectState(state.target_state.objclass,
                                            state.target_state["pos"])
        return cospomdp.ReducedState(
            self.robot_id, self.target_id,
            next_robot_state, target_state)

    def probability(self, next_state, state, action):
        expected_next_state = self.sample(state, action)
        return indicator(expected_next_state == next_state, epsilon=1e-12)


class HighLevelDetectionModel(cospomdp.DetectionModel):
    """High level detection model."""
    def __init__(self, detecting_class, true_pos_rate, rand=random):
        """
        Detector for detecting class.
        `true_pos_rate` is the true positive rate of detection.
        """
        self.detecting_class = detecting_class
        self._true_pos_rate = true_pos_rate
        self._rand = rand

    def probability(self, object_observation, object_state, robot_state):
        if object_observation.objclass != self.detecting_class\
           or object_state.objclass != self.detecting_class:
            return 1e-12

        if object_observation.location is None:
            # negative; didn't observe the object
            if object_state["pos"] == robot_state["pos"]:
                # robot expects to see the object, but did not. False negative
                return 1.0 - self._true_pos_rate
            else:
                # robot isn't expecting to see the object and it didn't.
                # because we are not considering false positives so it's 100%
                return 1.0
        else:
            confidence = 1.0
            if hasattr(object_observation, "confidence"):
                confidence = object_observation.confidence

            if object_observation.location == object_state["pos"]\
               and object_state["pos"] == robot_state["pos"]:
                return self._true_pos_rate * confidence
            else:
                # robot isn't expecting to see the object and it did.
                # because we are not considering false positives so it's 0%
                return 1e-12  # TODO: I am not sure about this.

    def sample(self, object_state, robot_state):
        if object_state["pos"] == robot_state["pos"]:
            # We expect the robot to be able to detect the object,
            # subject to class-specific uncertainty
            if self._true_pos_rate >= self._rand.uniform(0, 1):
                # True positive
                return cospomdp.ObjectObservation(object_state.objclass,
                                                  object_state["pos"])
            else:
                # False negative
                return cospomdp.ObjectObservation(object_state.objclass, None)
        else:
            # the robot is not at a location where the object
            # can be observed (of course, there could be multiple
            # locations where the robot can be to observe the target,
            # but because we will have a belief over locations to observe
            # target, this should not be a problem)
            #
            # we do not model false positives in the POMDP model at the high
            # level. It is subject to too many factors unmodelable.
            return cospomdp.ObjectObservation(object_state.objclass, None)


class HighLevelObservationModel(pomdp_py.ObservationModel):
    def __init__(self, target_class, detection_config, corr_dists, rand=random):
        """
        Args:
            detection_config (dict):  {"<object class>" : <true positive rate>}
            corr_dists (dict):  {"object class" for Si : Pr(Si | Starget) JointDist}
        """
        self._oms = {}
        self.detection_config = detection_config
        for objclass in detection_config:
            true_pos_rate = detection_config[objclass]
            detection_model = HighLevelDetectionModel(objclass, true_pos_rate, rand=rand)
            if objclass == target_class:
                corr_dist = None
            else:
                corr_dist = corr_dists[objclass]
            self._oms[objclass] = cospomdp.ObjectObservationModel(
                objclass, target_class, detection_model, corr_dist)

    def sample(self, next_state, action):
        return cospomdp.Observation(tuple(self._oms[objclass].sample(next_state, action)
                                          for objclass in self._oms))

    def probability(self, observation, next_state, action):
        return math.prod([self._oms[zi.objclass].probability(zi, next_state, action)
                          for zi in observation.object_observations
                          if zi.objclass in self.detection_config])

from ..probability import JointDist, Event, TabularDistribution
class HighLevelCorrelationDist(JointDist):
    def __init__(self, objclass, target_class, search_region, corr_func):
        """
        Models Pr(Si | Starget) = Pr(objclass | target_class)
        Args:
            objclass (str): class corresponding to the state variable Si
            target_class (str): target object class
            corr_func: can take in a target location, and an object location,
                and return a value, the greater, the more correlated.
        """
        self.objclass = objclass
        self.target_class = target_class
        self.search_region = search_region
        super().__init__([objclass, target_class])

        # calculate weights
        self.dists = {}  # maps from target state to
        for target_pos in search_region:
            target_state = HighLevelObjectState(target_class, target_pos)
            weights = {}
            for object_pos in search_region:
                object_state = HighLevelObjectState(objclass, object_pos)
                prob = corr_func(target_pos, object_pos,
                                 target_class, objclass)
                weights[Event({self.objclass: object_state})] = prob
            self.dists[target_state] =\
                TabularDistribution([self.objclass], weights, normalize=True)

    def marginal(self, variables, evidence):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence, i.e. evidence (if supplied);

        NOTE: Only supports variables = [objclass]
        with evidence being a specific target state

        variables (array-like);
        evidence (dict) mapping from variable name to value"""
        assert variables == [self.objclass],\
            "CorrelationDist can only be used to infer distribution"\
            "over the correlated object's state"
        assert self.target_class in evidence\
            and evidence[self.target_class].objclass == self.target_class,\
            "When inferring Pr(Si | Starget), you must provide a value for Starget"\
            "i.e. set evidence = <some target state>"
        target_state = evidence[self.target_class]
        if target_state not in self.dists:
            raise ValueError("Unexpected value for target state in evidence: {}".format(target_state))
        return self.dists[target_state]

    def valrange(self, var):
        if var != self.target_class and var != self.objclass:
            raise ValueError("Unable to return value range for {} because it is not modeled"\
                             .format(var))
        # For either object, the value range is the search region.
        return [HighLevelObjectState(var, pos)
                for pos in self.search_region]


class HighLevelPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, search_region, num_visits_init=10, val_init=constants.TOS_REWARD_HI):
        self.search_region = search_region
        self.action_prior = HighLevelPolicyModel.ActionPrior(num_visits_init, val_init, self)

    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state, history=None):
        return self._get_all_moves(state) + [SearchDecision(), DoneDecision()]

    def _get_all_moves(self, state):
        return [MoveDecision(dest)
                for dest in self.search_region.neighbor(state.robot_state["pos"])]

    def rollout(self, state, history=None):
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state), 1)[0]

    class ActionPrior(pomdp_py.ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            preferences = set()
            moves = self.policy_model._get_all_moves(state)
            target_pos = state.target_state["pos"]
            current_dist = euclidean_dist(state.robot_state["pos"], target_pos)
            for move in moves:
                if euclidean_dist(move.dest, target_pos) < current_dist:
                    preferences.add((move, self.num_visits_init, self.val_init))
            if state.robot_state["pos"] == target_pos:
                preferences.add((SearchDecision(), self.num_visits_init, self.val_init))
            return preferences

class HighLevelRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        if isinstance(action, DoneDecision):
            if next_state.robot_state["status"] == HighLevelStatus.DONE:
                return constants.TOS_REWARD_HI*2
            else:
                return constants.TOS_REWARD_LO*2
        elif isinstance(action, SearchDecision):
            if next_state.robot_state["pos"] == next_state.target_state["pos"]:
                return constants.TOS_REWARD_HI
            else:
                return constants.TOS_REWARD_LO
        else:
            return constants.TOS_REWARD_STEP


class ThorObjectSearchCOSPOMDP(POMDP):
    """The COSPOMDP for Thor Object Search;
    It is a high-level planning framework.

    State:
        robot state: 2D reachable position
        target state: 2D reachable position; Instead of meaning
            "the location of the target", means "the location the
             robot should be in if it wants to detect the target"
    Action (Decision):
        move_to: moves the robot to another reachable position
                 (by default, nearby)
        search: decides to search within a local region (i.e. where the robot is)
        done: declares victory.
    """
    def __init__(self,
                 task_config,
                 search_region,
                 init_robot_pos,
                 detection_config,
                 corr_dists,
                 planning_config,
                 init_target_belief="uniform"):
        """
        Args:
            task_config (dict): Common task configuration in thor
            detection_config (dict): See HighLevelObservationModel
            corr_dists (dict): Maps from object class to JointDist,
                that is, it includes all Pr(Si | Starget) distributions for all Si{1<=i<=N).
            init_robot_pos (tuple): initial robot position.
        """
        self.search_region = search_region
        init_robot_state = HighLevelRobotState(init_robot_pos, HighLevelStatus.INITIAL)
        init_robot_belief = pomdp_py.Histogram({init_robot_state : 1.0})

        self.robot_id = task_config.get("robot_id", "robot0")
        if task_config["task_type"] == "class":
            target_class = task_config["target"]
            target_id = target_class
        else:
            target_id = task_config["target"]
            target_class = thor_object_type(target_id)
        self.target_class = target_class
        self.target_id = target_id

        if init_target_belief == "uniform":
            init_target_belief = pomdp_py.Histogram(
                normalize({HighLevelObjectState(target_class, pos): 1.0
                           for pos in search_region}))

        init_belief = HighLevelOOBelief(
            self.robot_id, self.target_id,
            init_robot_belief, init_target_belief)
        transition_model = HighLevelTransitionModel(self.robot_id, self.target_id)
        observation_model = HighLevelObservationModel(target_class, detection_config, corr_dists)
        policy_model = HighLevelPolicyModel(search_region,
                                            **planning_config.get("action_prior_params", {}))
        reward_model = HighLevelRewardModel()
        super().__init__(init_belief, policy_model,
                         transition_model, observation_model, reward_model)
        self._planner = pomdp_py.POUCT(max_depth=planning_config["max_depth"],
                                       discount_factor=planning_config["discount_factor"],
                                       num_sims=planning_config["num_sims"],
                                       exploration_const=planning_config["exploration_const"],
                                       rollout_policy=policy_model,
                                       action_prior=policy_model.action_prior)

    def plan_step(self):
        return self._planner.plan(self)

    def update(self, action, tos_observation):
        """
        Args:
            action (Decision): decision made by high level planner
            tos_observation (TOS_Observation): image, depth image, detections.
        """
        next_robot_state = self.transition_model.sample(self.belief.mpe(), action).robot_state
        zobjs = []
        for xyxy, conf, cls in tos_observation.detections:
            zi = cospomdp.ObjectObservation(cls, next_robot_state["pos"])
            zobjs.append(zi)
        observation = cospomdp.Observation(tuple(zobjs))

        # update planner
        self._planner.update(self, action, observation)

        # update belief
        next_target_belief = {}
        for pos in self.search_region:
            target_state = HighLevelObjectState(self.target_class, pos)
            next_state = cospomdp.ReducedState(self.robot_id, self.target_id,
                                               next_robot_state, target_state)
            pr_o = self.observation_model.probability(observation, next_state, action)
            next_target_belief[target_state] = pr_o * self.belief[next_state]
        next_belief = HighLevelOOBelief(self.robot_id, self.target_id,
                                        pomdp_py.Histogram({next_robot_state : 1.0}),
                                        normalize(next_target_belief))
        self.set_belief(next_belief)

    def debug_last_plan(self):
        pomdp_py.print_preferred_actions(self.tree)
        # pomdp_py.print_tree(self.tree)
