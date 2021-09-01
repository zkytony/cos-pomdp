import random
import time
import math
import pomdp_py
import thortils as tt

import cospomdp
from cospomdp.utils.math import indicator, normalize, euclidean_dist
from cospomdp_apps.basic import PolicyModel2D, RobotTransition2D
from cospomdp_apps.basic.action import Move2D, ALL_MOVES_2D, Done

from .components.action import (grid_navigation_actions2d,
                                from_grid_action_to_thor_action_params)

from ..common import TOS_Action, ThorAgent
from .. import constants


class GridMapSearchRegion(cospomdp.SearchRegion2D):
    def __init__(self, grid_map):
        super().__init__(grid_map.obstacles)
        # self._obstacles = grid_map.obstacles


class ThorObjectSearchCosAgent(ThorAgent):

    @property
    def belief(self):
        return self.cos_agent.belief

    @property
    def robot_id(self):
        return self.cos_agent.robot_id

    @property
    def target_id(self):
        return self.cos_agent.target_id

    def sensor(self, objid):
        return self.cos_agent.sensor(objid)

    @property
    def detectable_objects(self):
        return self.cos_agent.detectable_objects

    def _build_detectors(self, detector_specs):
        detectable_objects = {}  # objects we care about are detectable objects
        detectors = {}
        for obj in self.task_config["detectables"]:
            if len(obj) == 2:
                object_id, object_class = obj
            else:
                object_id = object_class = obj
            detectable_objects[object_id] = (object_id, object_class)

            detector_type, sensor_params, quality_params = detector_specs[object_id]
            if detector_type.strip() == "fan-nofp":
                if type(sensor_params) == str:
                    sensor_params = eval(f"dict({sensor_params.strip()})")
                if quality_params == str:
                    quality_params = eval(quality_params.strip())
                detector = cospomdp.FanModelNoFP(object_id, sensor_params, quality_params)
                detectors[object_id] = detector
        return detectors, detectable_objects

    def _build_corr_dists(self, corr_specs, objects):
        target_id = self.target[0]
        corr_dists = {}
        for key in corr_specs:
            obj1, obj2 = key
            if obj1 == target_id:
                other = obj2
            elif obj2 == target_id:
                other = obj1
            else:
                continue

            if other not in corr_dists:
                corr_func, corr_func_args = corr_specs[key]
                if type(corr_func) == str:
                    corr_func = eval(corr_func)
                corr_dists[other] = cospomdp.CorrelationDist(objects[other],
                                                             objects[target_id],
                                                             self.search_region,
                                                             corr_func,
                                                             corr_func_args=corr_func_args)
        return corr_dists

    def interpret_robot_obz(tos_observation):
        raise NotImplementedError

    def update(self, tos_action, tos_observation):
        """
        Given TOS_Action and TOS_Observation, update the agent's belief, etc.
        """
        # Because policy_model's movements are already in sync with task movements,
        # we can directly get the POMDP action from there.
        objobzs = {}
        for cls in self.detectable_objects:
            objobzs[cls] = cospomdp.Loc(cls, None)

        for detection in tos_observation.detections:
            xyxy, conf, cls, loc3d = detection
            thor_x, _, thor_z = loc3d
            x, z = self.grid_map.to_grid_pos(thor_x, thor_z)
            objobzs[cls] = cospomdp.Loc(cls, (x, z))

        robotobz = self.interpret_robot_obz(tos_observation)
        observation = cospomdp.CosObservation(robotobz, objobzs)
        action = self.interpret_action(tos_action)
        self._update_belief(action, observation)



class ThorObjectSearchBasicCosAgent(ThorObjectSearchCosAgent):
    """
    The BasicCosAgent contains a COS-POMDP that is instantiated as in
    cospomdp_apps.basic; It uses a 2D grid map representation of the search
    region, and a 2D robot state representation. It directly outputs
    low-level movement actions such as MoveAhead; Therefore, there is
    no need for "goal interpretation" and the output is directly
    wrapped by a TOS_Action for execution.

    Note that this agent does not make use of LookUp / LookDown actions.
    That's why it is basic!
    """
    AGENT_USES_CONTROLLER = False
    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 solver,
                 solver_args,
                 grid_map,
                 thor_agent_pose,
                 thor_prior={}):
        """
        controller (ai2thor Controller)
        task_config (dict) configuration; see make_config
        corr_specs (dict): Maps from (target_id, object_id) to (corr_func, corr_func_args)
        detector_specs (dict): Maps from object_id to (detector_type, sensor_params, quality_params)
        solver (str): name of solver
        solver_args (dict): arguments for the solver
        thor_prior: dict mapping from thor location to probability; If empty, then the prior will be uniform.
        """

        robot_id = task_config['robot_id']
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.grid_map = grid_map
        self.search_region = search_region

        init_robot_pose = grid_map.to_grid_pose(
            thor_agent_pose[0][0],  #x
            thor_agent_pose[0][2],  #z
            thor_agent_pose[1][1]   #yaw
        )

        # TODO: SIMPLIFY - just use target_class
        if task_config["task_type"] == 'class':
            target_id = task_config['target']
            target_class = task_config['target']
            target = (target_id, target_class)
        else:
            target = task_config['target']  # (target_id, target_class)
            target_id = target[0]
        self.task_config = task_config
        self.target = target

        detectors, detectable_objects = self._build_detectors(detector_specs)
        corr_dists = self._build_corr_dists(corr_specs, detectable_objects)

        reward_model = cospomdp.ObjectSearchRewardModel(
            detectors[target_id].sensor,
            task_config["nav_config"]["goal_distance"] / grid_map.grid_size,
            robot_id, target_id,
            **task_config["reward_config"])

        # Construct CosAgent, the actual POMDP
        init_robot_state = cospomdp.RobotState2D(robot_id, init_robot_pose)
        robot_trans_model = RobotTransition2D(robot_id, reachable_positions)
        movement_params = self.task_config["nav_config"]["movement_params"]
        self.navigation_actions = grid_navigation_actions2d(movement_params, grid_map.grid_size)
        policy_model = PolicyModel2D(robot_trans_model, reward_model,
                                     movements=self.navigation_actions)
        prior = {grid_map.to_grid_pos(p[0], p[2]): thor_prior[p]
                 for p in thor_prior}
        self.cos_agent = cospomdp.CosAgent(target, init_robot_state,
                                           search_region, robot_trans_model, policy_model,
                                           corr_dists, detectors, reward_model,
                                           prior=prior)
        # construct solver
        if solver == "pomdp_py.POUCT":
            self.solver = pomdp_py.POUCT(**solver_args,
                                         rollout_policy=self.cos_agent.policy_model)
        else:
            self.solver = eval(solver)(**solver_args)


    def act(self):
        """
        Output a TOS_Action
        """
        action = self.solver.plan(self.cos_agent)

        # Need to return TOS_Action
        if not isinstance(action, TOS_Action):
            if isinstance(action, Move2D):
                name = action.name
                params = self.movement_params(action)
            elif isinstance(action, Done):
                name = "done"
                params = {}
            return TOS_Action(name, params)
        else:
            return action

    def movement_params(self, action):
        """Different from external agents, which uses deep learning models that
        are trained with fixed rotation sizes, here we are able to set the parameters
        of ai2thor movements based on POMDP action's parameters. So we make sure
        they are in sync - i.e. we convert parameters in `action` to parameters
        that can be used for ai2thor.
        """
        return from_grid_action_to_thor_action_params(action, self.grid_map.grid_size)

    def interpret_robot_obz(self, tos_observation):
        thor_robot_pose = tos_observation.robot_pose
        thor_robot_pose2d = (thor_robot_pose[0]['x'], thor_robot_pose[0]['z'], thor_robot_pose[1]['y'])
        robot_pose = self.grid_map.to_grid_pose(*thor_robot_pose2d)
        return cospomdp.RobotObservation(self.robot_id,
                                         robot_pose,
                                         cospomdp.RobotStatus(done=tos_observation.done))

    def interpret_action(self, tos_action):
        cospomdp_actions = {a.name: a for a in set(self.navigation_actions) | {Done()}}
        if tos_action.name in cospomdp_actions:
            action = cospomdp_actions[tos_action.name]
            return action
        else:
            raise ValueError("Cannot understand action {}".format(action))

    def _update_belief(self, action, observation):
        """
        Here, action, observation are already interpreted.
        """
        self.cos_agent.update(action, observation)
        self.solver.update(self.cos_agent, action, observation)