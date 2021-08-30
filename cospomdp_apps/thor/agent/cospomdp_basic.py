import random
import time
import math
import pomdp_py
import thortils as tt

from cospomdp.utils.math import indicator, normalize, euclidean_dist
from cospomdp import *
from cospomdp_apps.basic import PolicyModel2D, RobotTransition2D
from cospomdp_apps.basic.action import Move2D, ALL_MOVES_2D, Done

from ..common import TOS_Action, ThorAgent
from .. import constants


class GridMapSearchRegion(SearchRegion2D):
    def __init__(self, grid_map):
        super().__init__(grid_map.obstacles)
        # self._obstacles = grid_map.obstacles


class ThorObjectSearchBasicCosAgent(ThorAgent):
    """
    The BasicCosAgent contains a COS-POMDP that is instantiated as in
    cospomdp_apps.basic; It uses a 2D grid map representation of the search
    region, and a 2D robot state representation. It directly outputs
    low-level movement actions such as MoveAhead; Therefore, there is
    no need for "goal interpretation" and the output is directly
    wrapped by a TOS_Action for execution.
    """
    AGENT_USES_CONTROLLER = True
    def __init__(self,
                 controller,
                 task_config,
                 corr_specs,
                 detector_specs,
                 solver,
                 solver_args,
                 prior="uniform"):
        """
        controller (ai2thor Controller)
        task_config (dict) configuration; see make_config
        corr_specs (dict): Maps from (target_id, object_id) to (corr_func, corr_func_args)
        detector_specs (dict): Maps from object_id to (detector_type, sensor_params, quality_params)
        solver (str): name of solver
        solver_args (dict): arguments for the solver
        """

        robot_id = task_config['robot_id']
        grid_size = tt.thor_grid_size_from_controller(controller)
        grid_map = tt.convert_scene_to_grid_map(
            controller, tt.thor_scene_from_controller(controller), grid_size)
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.grid_map = grid_map
        self.search_region = search_region

        thor_robot_pose = tt.thor_agent_pose(controller.last_event)
        init_robot_pose = grid_map.to_grid_pose(
            thor_robot_pose[0]['x'], thor_robot_pose[0]['z'], thor_robot_pose[1]['y']
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

        reward_model = ObjectSearchRewardModel(
            detectors[target_id].sensor,
            task_config["nav_config"]["goal_distance"] / grid_size,
            robot_id, target_id)

        prior_dist = {}
        if prior == "informed":
            thor_x, _, thor_z = tt.thor_closest_object_of_type_position(controller, target_class, as_tuple=True)
            x, z = self.grid_map.to_grid_pos(thor_x, thor_z)
            prior_dist = {(x,z): 1e5}

        # Construct CosAgent, the actual POMDP
        init_robot_state = RobotState2D(robot_id, init_robot_pose)
        robot_trans_model = RobotTransition2D(robot_id, reachable_positions)
        policy_model = PolicyModel2D(robot_trans_model, reward_model)
        self.cos_agent = CosAgent(target, init_robot_state,
                                  search_region, robot_trans_model, policy_model,
                                  corr_dists, detectors, reward_model,
                                  prior=prior_dist)
        # construct solver
        if solver == "pomdp_py.POUCT":
            self.solver = pomdp_py.POUCT(**solver_args,
                                         rollout_policy=self.cos_agent.policy_model)
        else:
            self.solver = eval(solver)(**solver_args)


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
                detector = FanModelNoFP(object_id, sensor_params, quality_params)
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
                corr_dists[other] = CorrelationDist(objects[other],
                                                    objects[target_id],
                                                    self.search_region,
                                                    corr_func,
                                                    corr_func_args=corr_func_args)
        return corr_dists

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

    def act(self):
        """
        Output a TOS_Action
        """
        action = self.solver.plan(self.cos_agent)

        # Need to return TOS_Action
        if not isinstance(action, TOS_Action):
            if isinstance(action, Move2D):
                name = action.name
                params = self.movement_params(name)
            elif isinstance(action, Done):
                name = "done"
                params = {}
            return TOS_Action(name, params)
        else:
            return action

    def update(self, tos_action, tos_observation):
        """
        Given TOS_Action and TOS_Observation, update the agent's belief, etc.
        """
        action_names = {a.name: a for a in ALL_MOVES_2D | {Done()}}
        if tos_action.name in action_names:
            action = action_names[tos_action.name]
        else:
            raise ValueError("Cannot understand action {}".format(action))

        objobzs = {}
        for cls in self.detectable_objects:
            objobzs[cls] = Loc(cls, None)

        for detection in tos_observation.detections:
            xyxy, conf, cls, loc3d = detection
            thor_x, _, thor_z = loc3d
            x, z = self.grid_map.to_grid_pos(thor_x, thor_z)
            objobzs[cls] = Loc(cls, (x, z))

        thor_robot_pose = tos_observation.robot_pose
        thor_robot_pose2d = (thor_robot_pose[0]['x'], thor_robot_pose[0]['z'], thor_robot_pose[1]['y'])
        robot_pose = self.grid_map.to_grid_pose(*thor_robot_pose2d)
        # TODO: properly set status - right now there is only 'done' and it
        # doesn't affect behavior if this is always false because task success
        # depends on taking the done action, not the done status.
        status = RobotStatus()
        robotobz = RobotObservation(self.robot_id, robot_pose, status)
        observation = CosObservation(robotobz, objobzs)
        print(observation)
        self.cos_agent.update(action, observation)
        self.solver.update(self.cos_agent, action, observation)
