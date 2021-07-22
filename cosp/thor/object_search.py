import random
import math
import time
from collections import namedtuple

import pomdp_py
import ai2thor
import ai2thor.util.metrics as metrics

from thortils import (thor_camera_horizon,
                      thor_closest_object_of_type,
                      thor_object_in_fov,
                      thor_object_of_type_in_fov,
                      thor_object_pose,
                      thor_object_position,
                      thor_object_receptors,
                      thor_object_type,
                      thor_reachable_positions,
                      thor_agent_pose)
from thortils.vision import thor_img, thor_img_depth, thor_object_bboxes
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)
from thortils.utils import (to_degrees, closest,
                            normalize_angles, euclidean_dist)

from .result_types import PathResult, HistoryResult
from .utils import plot_path, plt, as_tuple
from .task import ThorEnv, ThorAgent
from . import constants


# ------------- Task ------------- #
class TOS(ThorEnv):
    """
    TOS is short for ThorObjectSearch task.
    This represents the environment of running a single object search task.
    """
    # State, Action, Observation used in object search task
    Action = namedtuple("Action", ['name', 'params'])
    State = namedtuple("State", ['agent_pose', 'horizon'])
    Observation = namedtuple("Observation", ["img", "img_depth", "bboxes"])

    def __init__(self, controller, task_config):
        """
        If task_type is "class", then target is an object type.
        If task_type is "object", then target is an object ID.
        """
        task_type = task_config["task_type"]
        target = task_config["target"]
        if task_type not in {"class", "object"}:
            raise ValueError("Invalid target type: {}".format(task_type))
        super().__init__(controller)
        self.target = target
        self.task_type = task_type
        self.goal_distance = task_config["nav_config"]["goal_distance"]
        self.task_config = task_config

    def compute_results(self):
        """
        We will compute:
        1. Discounted cumulative reward
           Will save the entire trace of history.

        2. SPL. Even though our problem involves open/close,
           the optimal path should be just the navigation length,
           because the agent just needs to navigate to the container, open it
           and then perhaps look down, which doesn't change the path length.
           This metric alone won't tell the full story. Because it obscures the
           cost of actions that don't change the robot's location. So a combination with
           discounted reward is better.

           Because SPL is a metric over all trials, we will return the
           result for individual trials, namely, the path length, shortest path length,
           and success

        Note: it appears that the shortest path from ai2thor isn't snapped to grid,
        or it skips many steps. That makes it slightly shorter than the path found by
        our optimal agent. But will still use it per
        """
        # Uses the ThorObjectSearchOptimalagent to compute the path as shortest path.
        plan, poses = ThorObjectSearchOptimalAgent.plan(
            self.controller, self.init_state.agent_pose,
            self.target, self.task_type,
            **self.task_config["nav_config"])
        shortest_path = [p[0] for p in poses]  # get positions only

        actual_path = self.get_current_path()
        last_reward = self._history[-1][-1]
        success = last_reward == constants.TOS_REWARD_HI
        return [PathResult(shortest_path, actual_path, success),
                HistoryResult(self._history)]

    def get_current_path(self):
        """Returns a list of dict(x=,y=,z=) positions,
        using the history up to now, for computing results"""
        path = []
        for tup in self._history:
            state = tup[0]
            x, y, z = state.agent_pose[0]
            agent_position = dict(x=x, y=y, z=z)
            path.append(agent_position)
        return path

    def get_observation(self, event):
        img = thor_img(event)
        img_depth = thor_img(event)
        bboxes = thor_img(event)
        return TOS.Observation(img, img_depth, bboxes)

    def get_state(self, event):
        # stores agent pose as tuple, for convenience.
        agent_pose = thor_agent_pose(event, as_tuple=True)
        horizon = thor_camera_horizon(event)
        return TOS.State(agent_pose, horizon)

    def get_reward(self, state, action, next_state):
        """We will use a sparse reward."""
        if self.done(action):
            if self.success(action,
                            agent_pose=state.agent_pose,
                            horizon=state.horizon):
                return constants.TOS_REWARD_HI
            else:
                return constants.TOS_REWARD_LO
        else:
            return constants.TOS_REWARD_STEP

    def done(self, action):
        """Returns true if  the task is over. The object search task is over when the
        agent took the 'Done' action.
        """
        return action.name == "Done"

    def success(self, action, agent_pose=None, horizon=None):
        """Returns True if the task is a success.
        The task is success if the agent takes 'Done' and
        (1) the target object is within the field of view.
        (2) the robot is close enough to the target.
        Note: uses self.controller to retrieve target object position."""
        if action.name != "Done":
            return False

        event = self.controller.step(action="Pass")
        backup_state = self.get_state(event)

        if agent_pose is not None:
            # Teleport to the given agent pose then evaluate
            position, rotation = agent_pose
            self.controller.step(action="Teleport",
                                 position=position,
                                 rotation=rotation,
                                 horizon=horizon)

        # Check if the target object is within the field of view
        event = self.controller.step(action="Pass")
        if self.task_type == "class":
            object_type = self.target
            in_fov = thor_object_of_type_in_fov(event, object_type)
            p = thor_closest_object_of_type(event, object_type)["position"]
            objpos = (p['x'], p['y'], p['z'])
        else:
            object_id = self.target
            in_fov = thor_object_of_type_in_fov(event, object_id)
            objpos = thor_object_position(event, object_id, as_tuple=True)

        agent_position = thor_agent_pose(event, as_tuple=True)[0]
        object_distance = euclidean_dist(objpos, agent_position)
        # allows 0.1 to account for small difference due to floating point instability
        close_enough = (object_distance - self.goal_distance) <= 1e-1
        success = in_fov and close_enough

        # Teleport back, if necessary (i.e. if agent_pose is provided)
        if agent_pose is not None:
            position, rotation = backup_state.agent_pose
            horizon = backup_state.horizon
            self.controller.step(action="Teleport",
                                 position=position,
                                 rotation=rotation,
                                 horizon=horizon)
        if not success:
            if not in_fov:
                print("Object not in field of view!")
            if not close_enough:
                print("Object not close enough! Minimum distance: {}; Actual distance: {}".\
                      format(self.goal_distance, object_distance))
        return success

    def get_step_info(self, step):
        sp, a, o, r = self._history[step]
        return "Step {}: {}, Action: {}, Reward: {}".format(sp.agent_pose[0], step, a.name, r)

# Class naming aliases
ThorObjectSearch = TOS


# ------------- Agent and Optimal Agent ------------- #
class ThorObjectSearchAgent(ThorAgent):
    AGENT_USES_CONTROLLER = False


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
            name, params = self.plan[self._index]
            if name.startswith("Open"):
                action = TOS.Action(name, {"objectId": params[0]})
            else:
                # TODO: should use the params in the action tuple.
                action = TOS.Action(name, self.movement_params[name])
            self._index += 1
        else:
            action = TOS.Action("Done", {})
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
        openable_receptors = thor_object_receptors(controller,
                                                   target_object["objectId"],
                                                   openable_only=True)
        openable_receptor_ids = set(r for r in openable_receptors if r["openable"])

        goal_objects = openable_receptors + [target_object]
        overall_poses, overall_plan = [], []
        position, rotation = start_position, start_rotation
        for obj in goal_objects:
            _start_time = time.time()
            poses, plan = get_shortest_path_to_object(
                controller, obj["objectId"],
                position, rotation,
                return_plan=True,
                **nav_config
            )
            if len(plan) > 0:
                print("plan found in {:.3f}s".format(time.time() - _start_time))
            else:
                raise ValueError("Plan not found to {}".format(obj["objectId"]))

            if obj["objectId"] in openable_receptor_ids:
                open_action = ("OpenObject", object_id)
                plan.append(open_action)
                poses.append(poses[-1])  # just so that both lists have same length

            overall_poses.extend(poses)
            overall_plan.extend(plan)
            position, rotation = as_tuple(poses[-1])  # in case navigation is needed between containers (should be rare)

        if plan is None:
            raise ValueError("Plan to {} not found".format(target))

        return overall_plan, overall_poses
