import random
import math
import pomdp_py
from thortils import (thor_closest_object_of_type,
                      thor_reachable_positions,
                      thor_agent_pose,
                      thor_object_with_id,
                      thor_object_receptors,
                      thor_object_type,
                      thor_object_position,
                      thor_scene_from_controller,
                      thor_grid_size_from_controller,
                      thor_closest_object_of_type_position,
                      thor_pose_as_tuple,
                      thor_pose_as_dict,
                      convert_scene_to_grid_map)

from .common import TOS_Action
from . import constants
from cospomdp.utils.math import indicator, normalize, euclidean_dist
from cospomdp_apps.thor.replay import ReplaySolver

# Used by optimal agent
import time
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)
from cospomdp.domain.state import RobotStatus, RobotState2D
from cospomdp.domain.observation import Loc, CosObservation, RobotObservation
from cospomdp_apps.basic import PolicyModel2D, RobotTransition2D
from cospomdp_apps.basic.action import Move2D, ALL_MOVES_2D, Done
from cospomdp import *

class ThorAgent:
    AGENT_USES_CONTROLLER = False

    def movement_params(self, move_name):
        """Returns the parameter dict used for ai2thor Controller.step
        for the given move_name"""
        return self.task_config["nav_config"]["movement_params"][move_name]

    @property
    def vision_detector(self):
        return None


######################### Optimal Agent ##################################
class ThorObjectSearchOptimalAgent(ThorAgent):
    """
    The optimal agent uses ai2thor's shortest path method
    to retrieve a path, and then follows this path by taking
    appropriate actions.
    """

    # Because this agent is not realistic, we permit it to have
    # access to the controller.
    AGENT_USES_CONTROLLER = True

    def __init__(self,
                 controller,
                 task_config):
        """Builds the agent and computes a plan"""
        super().__init__()
        self.controller = controller
        self.target = task_config["target"]
        self.task_type = task_config["task_type"]
        self.movement_params = task_config["nav_config"]["movement_params"]

        start_pose = thor_agent_pose(self.controller, as_tuple=True)
        overall_plan, overall_poses = ThorObjectSearchOptimalAgent.plan(
            self.controller, start_pose, self.target,
            self.task_type, **task_config["nav_config"])

        self.plan = overall_plan
        self._poses = overall_poses
        self._index = 0

    def act(self):
        """Returns action in plan"""
        if self._index < len(self.plan):
            action_name, action_params = self.plan[self._index]
            if action_name not in self.movement_params:
                action = TOS_Action(action_name, action_params)
            else:
                # TODO: should use the params in the action tuple.
                action = TOS_Action(action_name, self.movement_params[action_name])
            self._index += 1
        else:
            action = TOS_Action("Done", {})
        return action

    def update(self, action, observation):
        pass

    @classmethod
    def plan(cls, controller, start_pose, target, task_type, **nav_config):
        """
        Plans navigation + container opening actions
        where the navigation actions are movements along the shortest path
        found by A* on the grid map of environment.

        Args:
            controller (ai2thor.Controller)
            start_pose (tuple): position (x,y,z), rotation (pitch, yaw, roll).
                Both are tuples.
            target (str): Object type or ID
            task_type (str): "class" or "object"
            nav_config (dict): Navigation configurations
                (e.g. goal_distance, v_angles, h_angles, etc.), used in path finding.
                Refer to get_shortest_path_to_object for complete configuration.
        Returns:
            plan, poses

            plan: List of (action_name, params) tuples representing actions.
                The params are specific to the action; For movements, it's
                (forward, pitch, yaw). For opening, it is an object id.
            poses: List of (dict(x=,y=,z=), dict(x=,y=,z=)) (position, rotation) tuples.
                where each pose at index i corresponds to the pose resulting from taking
                action i.
        """
        start_position, start_rotation = start_pose
        if task_type == "class":
            object_type = target
            target_object = thor_closest_object_of_type(controller, object_type)
        else:
            target_object_id = target
            target_object = thor_object_with_id(controller, target_object_id)

        # Check if the target object is within openable receptacle. If so,
        # need to navigate to the receptacle, open it.
        # so receptors are also targets we want to navigate to
        ### NOTE: This method is hacky. It works for some objects, but has the following issues:
        ### 1. The robot doesn't know if it can successfully open the container
        ###    door (especially fridge).
        ### 2. An object, if placed very low, may be blocked by e.g. tabletop
        ###    and right now the robot cannot adjust its height to detect those
        ###    objects
        ### 3. (This is unrelated to this method, but it's an issue) For some
        ###    objects, their height seems wrong with respect to robot's height,
        ###    which causes the wrong pitch rotation to be computed
        ### THEREFORE, this optimal agent isn't complete; It returns a plan,
        ### which is approximately optimal (shortest), but it won't be able to
        ### execute it successfully 100% of the time because of the above issues.
        openable_receptors = thor_object_receptors(controller,
                                                   target_object["objectId"],
                                                   openable_only=True)
        openable_receptor_ids = set(r["objectId"] for r in openable_receptors if r["openable"])

        goal_objects = openable_receptors + [target_object]
        overall_poses, overall_plan = [], []
        position, rotation = start_position, start_rotation
        for obj in goal_objects:
            _start_time = time.time()
            _nav_config = dict(nav_config)
            poses, plan = get_shortest_path_to_object(
                controller, obj["objectId"],
                position, rotation,
                return_plan=True,
                **nav_config
            )
            if plan is not None:
                print("plan found in {:.3f}s".format(time.time() - _start_time))
            else:
                raise ValueError("Plan not found to {}".format(obj["objectId"]))

            if len(poses) == 0:
                # No action needed. The robot is at where it should be already.
                assert len(plan) == 0
                pose_to_extend = thor_pose_as_dict((position, rotation))
            else:
                pose_to_extend = poses[-1]
                # in case navigation is needed between containers (should be rare),
                # update the position and rotation for the next search.
                position, rotation = thor_pose_as_tuple(poses[-1])

            # Check if we need an open action
            if obj["objectId"] in openable_receptor_ids:
                open_action = ("OpenObject", dict(objectId=obj["objectId"]))
                plan.append(open_action)
                poses.append(pose_to_extend)  # just so that both lists have same length

            overall_poses.extend(poses)
            overall_plan.extend(plan)


        if plan is None:
            raise ValueError("Plan to {} not found".format(target))

        return overall_plan, overall_poses
#############################################################################

class GridMapSearchRegion(SearchRegion2D):
    def __init__(self, grid_map):
        super().__init__(grid_map.obstacles)
        # self._obstacles = grid_map.obstacles

class ThorObjectSearchCosAgent(ThorAgent):
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
        grid_size = thor_grid_size_from_controller(controller)
        grid_map = convert_scene_to_grid_map(
            controller, thor_scene_from_controller(controller), grid_size)
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.grid_map = grid_map
        self.search_region = search_region

        thor_robot_pose = thor_agent_pose(controller.last_event)
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
            thor_x, _, thor_z = thor_closest_object_of_type_position(controller, target_class, as_tuple=True)
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
