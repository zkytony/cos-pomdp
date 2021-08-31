import pomdp_py
from ..domain.state import ObjectState, RobotState, CosState
from ..utils.math import normalize
from .belief import CosJointBelief
from .transition_model import CosTransitionModel
from .observation_model import (CosObjectObservationModel,
                                CosObservationModel)
# from .policy_model import PolicyModel2D
from tqdm import tqdm

class CosAgent(pomdp_py.Agent):

    def __init__(self,
                 target,
                 init_robot_state,
                 search_region,
                 robot_trans_model,
                 policy_model,
                 corr_dists,
                 detectors,
                 reward_model,
                 belief_type="histogram",
                 prior={}):
        """
        Args:
            robot_id (any hashable)
            init_robot_pose (x, y, th). The coordinate convention is (0,0) is bottom-left,
                with +x as th=0 and th increases counterclockwise.
            target (tuple): target_id, target_class
            search_region (SearchRegion2D): possible locations for the target
            reachable_positions (list): List of 2D (x,y) locations that are
                possible locations for the robot to reach
            corr_dists: Maps from objid to CorrelationDist Pr(si | starget)
                Does not need to contain an entry for the target object
            detectors: Maps from objid to a DetectionModel Pr(zi | si, srobot')
                Must contain an entry for the target object
            belief_type: type of belief representation.
            prior: Maps from search region location to a float.
            rollout_policy_model (RolloutPolicy)
        """
        self.search_region = search_region
        robot_id = init_robot_state.id
        target_id, target_class = target
        init_btarget = initialize_target_belief(target, search_region,
                                                belief_type, prior)
        init_brobot = initialize_robot_belief(init_robot_state)
        init_belief = CosJointBelief({robot_id: init_brobot,
                                      target_id: init_btarget})
        self.init_robot_state = init_robot_state

        transition_model = CosTransitionModel(target_id, robot_trans_model)

        omodels = {}
        for objid in detectors:
            if objid == target_id:
                omodel_i = CosObjectObservationModel(
                    target_id, target_id,
                    robot_id, detectors[objid])
            else:
                omodel_i = CosObjectObservationModel(
                    objid, target_id,
                    robot_id, detectors[objid], corr_dist=corr_dists[objid])
            omodels[objid] = omodel_i
        observation_model = CosObservationModel(robot_id, target_id, omodels)

        super().__init__(init_belief, policy_model,
                         transition_model, observation_model, reward_model)

    def sensor(self, objid):
        return self.observation_model.zi_models[objid].detection_model.sensor

    @property
    def detectable_objects(self):
        return set(self.observation_model.zi_models.keys())

    @property
    def target_id(self):
        return self.observation_model.target_id

    @property
    def robot_id(self):
        return self.observation_model.robot_id

    def update(self, action, observation):
        robotobz = observation.z(self.robot_id)
        rstate_class = self.init_robot_state.__class__
        next_srobot = rstate_class.from_obz(robotobz)
        new_brobot = pomdp_py.Histogram({next_srobot: 1.0})
        new_btarget = update_target_belief(
            self.target_id, self.belief.b(self.target_id), next_srobot,
            observation, self.observation_model)
        new_belief = CosJointBelief({self.robot_id: new_brobot,
                                     self.target_id: new_btarget})
        self.set_belief(new_belief)


def initialize_robot_belief(init_robot_state):
    """The robot state is known"""
    return pomdp_py.Histogram({init_robot_state: 1.0})

def initialize_target_belief(target, search_region, belief_type, prior):
    def _prob(prior, loc):
        return prior.get(loc, 1.0)

    target_id, target_class = target
    if belief_type == "histogram":
        hist = normalize({
            ObjectState(target_id, target_class, loc): _prob(prior, loc)
            for loc in search_region.locations
        })
        return pomdp_py.Histogram(hist)

    else:
        raise NotImplementedError("belief_type {} is not yet implemented".format(belief_type))

def update_target_belief(target_id, current_btarget, next_srobot,
                         observation, observation_model):
    """
    current_btarget: current target belief
    srobot: robot state corresponding to the observation.
    """
    if isinstance(current_btarget, pomdp_py.Histogram):
        new_btarget_hist = {}
        for starget in tqdm(current_btarget):
            state = CosState({target_id: starget,
                              next_srobot.id: next_srobot})
            pr_z = observation_model.probability(observation, state)
            new_btarget_hist[starget] = pr_z * current_btarget[starget]
        new_btarget = pomdp_py.Histogram(normalize(new_btarget_hist))
    return new_btarget
