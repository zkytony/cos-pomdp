import random
import math
from thortils import (thor_closest_object_of_type,
                      thor_reachable_positions,
                      thor_agent_pose,
                      thor_object_with_id,
                      thor_object_receptors,
                      thor_object_type,
                      thor_object_position,
                      thor_scene_from_controller,
                      thor_grid_size_from_controller,
                      thor_pose_as_tuple,
                      thor_pose_as_dict,
                      convert_scene_to_grid_map)

from .common import TOS_Action
from . import constants
from cospomdp.utils.math import indicator, normalize, euclidean_dist

# Used by optimal agent
import time
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)
from cospomdp.domain.action import Move2D
from cospomdp import *

class ThorAgent:
    AGENT_USES_CONTROLLER = False
    @property
    def detector(self):
        return None

class ThorPOMDPAgent(ThorAgent):
    def __init__(self, grid_map):
        # the grid map used to define POMDP state space
        self.grid_map


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
        super().__init__(grid_map.free_locations)
        self._obstacles = grid_map.obstacles

class ThorObjectSearchCosAgent(ThorAgent):
    def __init__(self,
                 controller,
                 task_config,
                 corr_specs,
                 detector_specs):

        robot_id = task_config['robot_id']
        thor_config = task_config['thor']
        scene = thor_config['scene']
        grid_map = convert_scene_to_grid_map(
            controller, scene,
            thor_config["GRID_SIZE"])

        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations

        thor_robot_pose = thor_agent_pose(controller.last_event)
        init_robot_pose = grid_map.to_grid_pose(
            thor_robot_pose[0]['x'], thor_robot_pose[0]['z'], thor_robot_pose[1]['y']
        )

        if task_config["task_type"] == 'class':
            target_id = target
            target_class = target
            target = (target_id, target_class)
        else:
            target = task_config['target']  # (target_id, target_class)

        detectable_objects = task_config["detectables"]  # [(object_id, object_class)]
        detectors = {}
        for obj in detectable_objects:
            if len(obj) == 2:
                object_id, object_class = obj
            else:
                object_id = object_class = obj
            detector_type, sensor_params, quality_params = detector_specs[object_id]
            if detector_type.strip() == "fan-nofp":
                if type(sensor_params) == str:
                    sensor_params = eval(f"dict({sensor_params.strip()})")
                if quality_params == str:
                    quality_params = eval(quality_params.strip())
                detector = FanModelNoFP(object_id, sensor_params, quality_params)
                detectors[object_id] = detector

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
                corr_func = eval(corr_specs[key][0])
                corr_func_args = corr_specs[key][1]
                corr_dists[other] = CorrelationDist(detectable_objects[other],
                                                    detectable_objects[target_id])

        reward_model = ObjectSearchRewardModel2D(
            detectors[target_id].sensor, thor_config["GOAL_DISTANCE"], robot_id, target)

        self.cos_agent = CosAgent(robot_id, init_robot_pose, target,
                                  search_region, reachable_positions,
                                  corr_dists, detectors, reward_model)
