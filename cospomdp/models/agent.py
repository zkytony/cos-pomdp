import pomdp_py
from ..domain.state import ObjectState2D, RobotState2D, CosState2D
from ..utils.math import normalize
from .belief import CosJointBelief
from .transition_model import RobotTransition2D, CosTransitionModel2D
from .observation_model import (CosObjectObservationModel2D,
                                CosObservationModel2D)
from .policy_model import PolicyModel2D
from tqdm import tqdm

class CosAgent(pomdp_py.Agent):

    def __init__(self, robot_id, init_robot_pose, target,
                 search_region, reachable_positions,
                 corr_dists, detectors, reward_model, belief_type="histogram", prior={}):
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
        """
        self.search_region = search_region
        target_id, target_class = target
        init_btarget = initialize_target_belief(target, search_region,
                                                belief_type, prior)
        init_brobot = initialize_robot_belief(robot_id, init_robot_pose)
        init_belief = CosJointBelief({robot_id: init_brobot,
                                      target_id: init_btarget})

        robot_trans_model = RobotTransition2D(robot_id, reachable_positions)
        transition_model = CosTransitionModel2D(target_id, robot_trans_model)

        omodels = {}
        for objid in detectors:
            if objid == target_id:
                omodel_i = CosObjectObservationModel2D(
                    target_id, target_id,
                    robot_id, detectors[objid])
            else:
                omodel_i = CosObjectObservationModel2D(
                    objid, target_id,
                    robot_id, detectors[objid], corr_dist=corr_dists[objid])
            omodels[objid] = omodel_i
        observation_model = CosObservationModel2D(robot_id, target_id, omodels)

        policy_model = PolicyModel2D(robot_trans_model, reward_model)
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
        new_brobot = pomdp_py.Histogram({robotobz.to_state(): 1.0})
        new_btarget = update_target_belief(
            self.target_id, self.belief.b(self.target_id), observation, self.observation_model)
        new_belief = CosJointBelief({self.robot_id: new_brobot,
                                     self.target_id: new_btarget})
        self.set_belief(new_belief)


def initialize_robot_belief(robot_id, init_robot_pose):
    init_robot_state = RobotState2D(robot_id, init_robot_pose)
    return pomdp_py.Histogram({init_robot_state: 1.0})

def initialize_target_belief(target, search_region, belief_type, prior):
    def _prob(prior, loc):
        return prior.get(loc, 1.0)

    target_id, target_class = target
    if belief_type == "histogram":
        hist = normalize({
            ObjectState2D(target_id, target_class, loc): _prob(prior, loc)
            for loc in search_region.locations
        })
        return pomdp_py.Histogram(hist)

    else:
        raise NotImplementedError("belief_type {} is not yet implemented".format(belief_type))

def update_target_belief(target_id, current_btarget, observation, observation_model):
    """
    current_btarget: current target belief
    """
    srobot = observation.z_robot.to_state()

    if isinstance(current_btarget, pomdp_py.Histogram):
        new_btarget_hist = {}
        for starget in tqdm(current_btarget):
            state = CosState2D({target_id: starget,
                                srobot.id: srobot})
            new_btarget_hist[starget] =\
                observation_model.probability(observation, state) * current_btarget[starget]
        new_btarget = pomdp_py.Histogram(normalize(new_btarget_hist))
    return new_btarget
