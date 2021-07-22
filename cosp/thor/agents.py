import random
import math
import time
from collections import namedtuple

from thortils import (thor_agent_pose,
                      thor_closest_object_of_type,
                      thor_object_pose,
                      thor_object_type,
                      thor_reachable_positions,
                      thor_object_receptors)
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)
from thortils.utils import (to_degrees, closest,
                            normalize_angles, euclidean_dist)
from .utils import plot_path, plt, as_tuple
from .task import TOS
from ..framework import Agent

class ThorAgent(Agent):
    def __init__(self):
        pass

    def act(self):
        pass

    def update(self, observation, reward):
        pass


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
        self.movement_params = task_config["movement_params"]

        start_pose = task_config.get(
            "start_pose", thor_agent_pose(self.controller, as_tuple=True))
        start_position, start_rotation = start_pose

        if self.task_type == "class":
            object_type = self.target
            target_object = thor_closest_object_of_type(self.controller, object_type)
        else:
            target_object_id = self.target
            target_object = thor_object_with_id(self.controller, target_object_id)

        # Check if the target object is within openable receptacle. If so,
        # need to navigate to the receptacle, open it.
        # so receptors are also targets we want to navigate to
        openable_receptors = thor_object_receptors(self.controller,
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
                **task_config
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
            raise ValueError("Plan to {} not found".format(self.target))
        self.plan = overall_plan
        self._poses = overall_poses
        self._index = 0

    def act(self):
        """Returns action in plan"""
        if self._index < len(self.plan):
            name = self.plan[self._index][0]
            if name.startswith("Open"):
                action = TOS.Action(name, {"objectId": self.plan})
            else:
                action = TOS.Action(name, self.movement_params[name])
            self._index += 1
        else:
            action = TOS.Action("Done", {})
        return action

    def update(self, action, observation):
        pass

    # @classmethod
    # def plan(cls, start_pose,
