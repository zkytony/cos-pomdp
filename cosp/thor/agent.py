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

from .common import TOS_Action, ThorAgent
from . import constants
from ..utils.math import indicator, normalize, euclidean_dist

# Used by optimal agent
import time
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)


# # Used by COSPOMDP agent
# from ..planning.hierarchical import HierarchicalPlanningAgent
# from .high_level import (HighLevelSearchRegion,
#                          HighLevelCorrelationDist,
#                          ThorObjectSearchCOSPOMDP)
# from .decisions import MoveDecision
# from .low_level import (LowLevelObjectState,
#                         MoveAction)
# from .common import ThorAgent
# from ..models.fansensor import FanSensor
# from thortils.navigation import (get_navigation_actions)


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


import pomdp_py
from ..models.state import ObjectState2D, JointState2D
from ..models.belief import LocBelief2D, JointBelief2D
from ..models.search_region import SearchRegion2D
from ..models.transition import RobotTransition2D, JointTransitionModel2D
from ..models.correlation import CorrelationDist
from ..models.observation import (CorrObservationModel,
                                  JointObservationModel,
                                  FanModelNoFP,
                                  ObjectDetection2D,
                                  JointObservation)
from ..models.policy import ThorPolicyModel2D
from ..models.reward import ThorRewardModel2D
from ..models.action import Move
from ..utils.math import roundany
from thortils.scene import convert_scene_to_grid_map
from thortils.navigation import convert_movement_to_action

class ThorObjectSearchCOSPOMDPAgent(pomdp_py.Agent, ThorAgent):
    AGENT_USES_CONTROLLER = True

    def __init__(self, controller,
                 task_config, detector_config,
                 corr_func, planning_config):
        self.task_config = task_config
        self.task_type = self.task_config["task_type"]
        if self.task_type == "class":
            self.target_class = self.task_config["target"]

        grid_size = thor_grid_size_from_controller(controller)
        reachable_positions = thor_reachable_positions(controller)
        scene = thor_scene_from_controller(controller)
        self.grid_map = convert_scene_to_grid_map(reachable_positions,
                                                  scene, grid_size)
        # initial belief
        search_region = SearchRegion2D({(x,y)
                                        for x in range(self.grid_map.width)
                                        for y in range(self.grid_map.length)})
        if task_config["prior"] == "uniform":
            init_target_belief = LocBelief2D.uniform(self.target_class, search_region)
        elif task_config["prior"] == "informed":
            ##informed
            obj = thor_closest_object_of_type(controller.last_event, self.target_class)
            thor_x, _, thor_z = thor_object_position(controller.last_event, obj["objectId"], as_tuple=True)
            x, z = self.grid_map.to_grid_pos(thor_x, thor_z)
            init_target_belief = LocBelief2D.informed(self.target_class,
                                                      (x, z),
                                                      search_region)
        else:
            raise ValueError("Invalid prior type")

        self.robot_id = task_config.get("robot_id", "robot0")
        robot_pose = thor_agent_pose(controller, as_tuple=True)
        thor_x, _, thor_z = robot_pose[0]
        _, yaw, _ = robot_pose[1]
        x, y = self.grid_map.to_grid_pos(thor_x, thor_z)
        init_robot_pose = (x, y, yaw)  # yaw-90 because that's how GridMap's angles match with Ai2Thor angle (HARD BUG!)
        init_robot_state = ObjectState2D(self.robot_id, dict(pose=init_robot_pose))
        init_robot_belief = pomdp_py.Histogram({init_robot_state : 1.0})
        init_belief = JointBelief2D(self.robot_id, self.target_class,
                                    init_robot_belief, init_target_belief)

        # Transition model
        robot_trans_model = RobotTransition2D(self.robot_id,
                                              self.grid_map.free_locations,
                                              diagonal_ok=constants.DIAG_MOVE)
        transition_model = JointTransitionModel2D(self.target_class,
                                                  robot_trans_model)

        # Observation model
        zi_models = {}
        for detectable_class in detector_config:
            dcfg = detector_config[detectable_class]
            corr_dist = CorrelationDist(detectable_class, self.target_class,
                                        search_region, corr_func)
            model_type = dcfg["type"]
            params = dcfg["params"]
            detector = eval(model_type)(**params)
            corr_model = CorrObservationModel(detectable_class, self.target_class,
                                              detector, corr_dist)
            zi_models[detectable_class] = corr_model
        observation_model = JointObservationModel(self.target_class, zi_models)

        # Reward model
        reward_model = ThorRewardModel2D(zi_models[self.target_class].detection_model.sensor)

        # Policy model
        policy_model = ThorPolicyModel2D(robot_trans_model, reward_model,
                                         num_visits_init=10,
                                         val_init=constants.TOS_REWARD_HI)
        super().__init__(init_belief, policy_model,
                         transition_model, observation_model, reward_model)

        self._planner = pomdp_py.POUCT(max_depth=planning_config["max_depth"],
                                       discount_factor=planning_config["discount_factor"],
                                       num_sims=planning_config["num_sims"],
                                       exploration_const=planning_config["exploration_const"],
                                       rollout_policy=policy_model,
                                       action_prior=policy_model.action_prior)

    def act(self):
        action = self._planner.plan(self)
        if isinstance(action, Move):
            movement_params = self.task_config["nav_config"]["movement_params"]
            return TOS_Action(action.name,
                              movement_params[action.name])
        return action

    def update(self, tos_action, observation):
        tos_observation, reward = observation

        if tos_action.name in self.task_config["nav_config"]["movement_params"]:
            movement, delta = convert_movement_to_action(
                tos_action.name, {tos_action.name : tos_action.params})
            forward, h_angle, v_angle = delta
            forward /= self.grid_map.grid_size  # because we consider GridMap coordinates
            # h_angle = -h_angle   # due to GridMap axes (HARD BUG!)
            # import pdb; pdb.set_trace()
            action = Move(movement, (forward, h_angle))  # We only care about 2D at this level.
        else:
            action = tos_action

        # update robot state
        mpe_state = self.belief.mpe()
        next_robot_state = self.transition_model.sample(mpe_state, action).robot_state
        next_robot_belief = pomdp_py.Histogram({next_robot_state : 1.0})
        print(next_robot_state)

        # get POMDP observation
        cls_to_loc3d = {}
        for xyxy, conf, cls, loc3d in tos_observation.detections:
            cls_to_loc3d[cls] = loc3d
        pomdp_detections = []
        for detectable_class in self.observation_model.classes:
            if detectable_class in cls_to_loc3d:
                loc3d = cls_to_loc3d[detectable_class]
                gx, gy = self.grid_map.to_grid_pos(loc3d[0], loc3d[2])
                zi = ObjectDetection2D(detectable_class, (gx, gy))
            else:
                zi = ObjectDetection2D(detectable_class, None)
            pomdp_detections.append(zi)
        observation = JointObservation(tuple(pomdp_detections))

        next_target_hist = {}
        target_belief = self.belief.target_belief
        for starget in target_belief:
            next_state = JointState2D(self.robot_id, self.target_class,
                                      {self.robot_id: next_robot_state,
                                       self.target_class: starget})
            next_target_hist[starget] =\
                self.observation_model.probability(observation, next_state, action) * target_belief[starget]
        next_target_belief = LocBelief2D(normalize(next_target_hist))
        if math.isnan(next_target_belief[next_target_belief.mpe()]):
            import pdb; pdb.set_trace()
        next_belief = JointBelief2D(self.robot_id, self.target_class,
                                    next_robot_belief, next_target_belief)
        self.set_belief(next_belief)

        self.debug_last_plan()

        # Update planner
        self._planner.update(self, action, observation)

    def debug_last_plan(self):
        pomdp_py.print_preferred_actions(self.tree)
        self.tree.print_children_value()
        # pomdp_py.print_tree(self.tree)







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


def thor_map_coordinates2D(reachable_positions, scene_name, grid_size):
    """Returns an array of 2D thor coordinates that includes
    both reachable and unreachable locations (essentially based on
    the rectangle that captures reachable_positions.)"""
    grid_map = convert_scene_to_grid_map(reachable_positions,
                                         scene_name, grid_size)
    coords = [grid_map.to_thor_pos(x, y)
              for x, y in grid_map.free_locations]
    return coords
