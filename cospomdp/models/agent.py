import pomdp_py
import random
from ..domain.state import ObjectState, ObjectState3D, RobotState, CosState
from ..utils.math import normalize, euclidean_dist
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
                 target_belief_initializer,
                 target_belief_updater,
                 belief_type="histogram",
                 use_heuristic=True,
                 bu_args={},
                 prior={}):
        """
        Args:
            robot_id (any hashable)
            init_robot_pose (x, y, th). The coordinate convention is (0,0) is bottom-left,
                with +x as th=0 and th increases counterclockwise.
            target (tuple): target_id, target_class
            search_region (SearchRegion2D): possible locations for the target
            corr_dists: Maps from objid to CorrelationDist Pr(si | starget)
                Does not need to contain an entry for the target object
            detectors: Maps from objid to a DetectionModel Pr(zi | si, srobot')
                Must contain an entry for the target object
            belief_type: type of belief representation.
                histogram or histogram-approx; If it is histogram, the belief
                is updated exactly. Otherwise, it is approximated; the belief
                at a fixed number of samples of places are updated, and the belief
                at other places are approximated by nearby places.
            bu_args (dict): Arguments for belief update; useful for approximate update.
            prior: Maps from search region location to a float.
            rollout_policy_model (RolloutPolicy)
        """
        self.search_region = search_region
        robot_id = init_robot_state.id
        target_id, target_class = target
        self._target = target
        self._belief_type = belief_type
        self._bu_args = bu_args
        init_btarget = target_belief_initializer(target, search_region,
                                                 belief_type, prior)
        init_brobot = self._initialize_robot_belief(init_robot_state)
        init_belief = CosJointBelief({robot_id: init_brobot,
                                      target_id: init_btarget})
        self.init_robot_state = init_robot_state
        self._target_belief_updater = target_belief_updater

        transition_model = CosTransitionModel(target_id, robot_trans_model)
        observation_model = build_cos_observation_model(corr_dists, detectors,
                                                        robot_id, target_id)

        policy_model.set_observation_model(observation_model,
                                           use_heuristic=use_heuristic)

        super().__init__(init_belief, policy_model,
                         transition_model, observation_model, reward_model)

    def sensor(self, objid):
        return self.observation_model.zi_models[objid].detection_model.sensor

    @property
    def detectable_objects(self):
        return set(self.observation_model.zi_models.keys())

    @property
    def target_id(self):
        return self._target[0]

    @property
    def target_class(self):
        return self._target[1]

    @property
    def robot_id(self):
        return self.observation_model.robot_id

    def update(self, action, observation):
        robotobz = observation.z(self.robot_id)
        rstate_class = self.init_robot_state.__class__
        next_srobot = rstate_class.from_obz(robotobz)
        new_brobot = pomdp_py.Histogram({next_srobot: 1.0})
        new_btarget = self._target_belief_updater(
            self.belief.b(self.target_id), next_srobot,
            observation, self.observation_model, self._belief_type, self._bu_args)
        new_belief = CosJointBelief({self.robot_id: new_brobot,
                                     self.target_id: new_btarget})
        self.set_belief(new_belief)

    def _initialize_robot_belief(self, init_robot_state):
        """The robot state is known"""
        return pomdp_py.Histogram({init_robot_state: 1.0})


def build_cos_observation_model(corr_dists, detectors, robot_id, target_id):
    """Construct CosObservationModel"""
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
    return observation_model
