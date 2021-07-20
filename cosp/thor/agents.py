import random
import math
from collections import namedtuple

from thortils import (thor_agent_pose,
                      thor_closest_object_of_type,
                      thor_object_pose,
                      thor_reachable_positions)
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)
from thortils.utils import (to_degrees, closest,
                            normalize_angles, euclidean_dist)
from .utils import plot_path, plt
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

        start_pose = thor_agent_pose(self.controller, as_tuple=True)
        start_position, start_rotation = start_pose

        if self.task_type == "class":
            get_path_func = get_shortest_path_to_object_type
        else:
            get_path_func = get_shortest_path_to_object

        poses, plan = get_path_func(
            controller, self.target,
            start_position, start_rotation,
            return_plan=True,
            **task_config)

        if plan is None:
            raise ValueError("Plan to {} not found".format(self.target))
        self.plan = plan
        self._poses = poses
        self._index = 0

    def act(self):
        """Returns action in plan"""
        if self._index < len(self.plan):
            name = self.plan[self._index][0]
            params = self.movement_params[name]
            action = TOS.Action(name, params)
            self._index += 1
        else:
            action = TOS.Action("Done", {})
        return action

    def update(self, action, observation):
        pass
