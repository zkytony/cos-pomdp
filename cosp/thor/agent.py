import random
import math
from thortils import (thor_closest_object_of_type,
                      thor_reachable_positions,
                      thor_agent_pose,
                      thor_object_with_id,
                      thor_object_receptors,
                      thor_object_type,
                      thor_scene_from_controller,
                      thor_grid_size_from_controller,
                      thor_pose_as_tuple,
                      thor_pose_as_dict,
                      convert_scene_to_grid_map)

from .common import TOS_Action, ThorAgent
from . import constants
from ..utils.math import indicator, normalize, euclidean_dist

# Used by optimal agent
import time
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)


# Used by COSPOMDP agent
from ..planning.hierarchical import HierarchicalPlanningAgent
from .high_level import (HighLevelSearchRegion,
                         HighLevelCorrelationDist,
                         ThorObjectSearchCOSPOMDP)
from .decisions import MoveDecision
from .low_level import (LowLevelObjectState,
                        MoveAction)
from .common import ThorAgent
from ..models.fansensor import FanSensor
from thortils.navigation import (get_navigation_actions)


class ThorObjectSearchAgent(ThorAgent):
    AGENT_USES_CONTROLLER = False

######################### Optimal Agent ##################################
class ThorObjectSearchOptimalAgent(ThorObjectSearchAgent):
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


# ###################### ThorObjectSearchCOSPOMDPAgent ########################
# class ThorObjectSearchCOSPOMDPAgent(HierarchicalPlanningAgent, ThorAgent):
#     # In the current version, uses the controller in order to determine
#     # the search region and initial robot pose. In the more general case,
#     # the agent will begin with a partial map, or no map at all, and needs
#     # to explore and expand the map; in that case controller is not needed.
#     AGENT_USES_CONTROLLER = True

#     def __init__(self, controller,
#                  task_config, detector_config,
#                  corr_func, planning_configs):
#         search_region = HighLevelSearchRegion(
#             thor_reachable_positions(controller))
#         robot_pose = thor_agent_pose(controller, as_tuple=True)
#         x, _, z = robot_pose[0]
#         init_robot_pos = (x, z)  # high-level position
#         self.init_robot_pose = robot_pose

#         self.task_config = task_config
#         self.task_type = self.task_config["task_type"]
#         if self.task_type == "class":
#             self.target_class = self.task_config["target"]

#         # coords2D = thor_map_coordinates2D(search_region.reachable_positions,
#         #                                   thor_scene_from_controller(controller),
#         #                                   thor_grid_size_from_controller(controller))
#         # self.target_belief = self._init_target_belief2D(coords2D, prior="uniform")
#         # self.camera_model = FanSensor(**detector_config["intrinsics"])

#         corr_dists = {
#             objclass: HighLevelCorrelationDist(objclass, self.target_class,
#                                                search_region, corr_func)
#             for objclass in detector_config["detection_rates"]
#             if objclass != self.target_class}

#         self.planning_configs = planning_configs
#         high_level_pomdp = ThorObjectSearchCOSPOMDP(
#             self.task_config,
#             search_region,
#             init_robot_pos,
#             detector_config["detection_rates"],
#             corr_dists,
#             self.planning_configs["high_level"])
#         super().__init__(high_level_pomdp)

#     @property
#     def robot_id(self):
#         return self.task_config["robot_id"]

#     def _decision_made(self, decision):
#         """
#         Prepares necessary arguments to build low-level POMDP
#         """
#         if isinstance(decision, MoveDecision):
#             movement_params = self.task_config["nav_config"]["movement_params"]
#             action_tuples = get_navigation_actions(movement_params=movement_params)
#             move_actions = [MoveAction(name, delta)
#                             for name, delta in action_tuples]
#             if self.low_level_pomdp is None:
#                 # This is the first time to create a low level pomdp;
#                 robot_pose = self.init_robot_pose
#             else:
#                 # The low_level_pomdp's belief should always contain robot pose
#                 robot_pose = self.low_level_belief.mpe().robot_state["pose"]
#             return dict(robot_id=self.robot_id,
#                         move_actions=move_actions,
#                         robot_pose=robot_pose,
#                         planning_config=self.planning_configs["MoveDecision"])

#     def _action_computed(self, pomdp_action):
#         """Converts an Action to TOS_Action which can be executed
#         in Ai2Thor."""
#         if isinstance(pomdp_action, MoveAction):
#             movement_params = self.task_config["nav_config"]["movement_params"]
#             return TOS_Action(pomdp_action.name,
#                               movement_params[pomdp_action.name])

#     def _init_target_belief2D(self, coords, prior="uniform"):
#         if prior == "uniform":
#             return normalize({LowLevelObjectState(self.target_class, {"pos": pos}) : 1.0
#                               for pos in coords})
#         raise NotImplementedError

#     def update(self, action, observation):
#         """Update belief given action and observation (which is
#         actually a (reward, obseravtion) tuple)"""
#         observation, reward = observation
#         import pdb; pdb.set_trace()
#         # Update low-level target belief.


# def thor_map_coordinates2D(reachable_positions, scene_name, grid_size):
#     """Returns an array of 2D thor coordinates that includes
#     both reachable and unreachable locations (essentially based on
#     the rectangle that captures reachable_positions.)"""
#     grid_map = convert_scene_to_grid_map(reachable_positions,
#                                          scene_name, grid_size)
#     coords = [grid_map.to_thor_pos(x, y)
#               for x, y in grid_map.free_locations]
#     return coords
