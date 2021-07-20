import random
import math
from collections import namedtuple
from ai2thor.util.metrics import (get_shortest_path_to_object,
                                  get_shortest_path_to_object_type)

from thortils import (thor_agent_pose,
                      thor_closest_object_of_type,
                      thor_object_pose,
                      thor_reachable_positions)
from thortils.navigation import find_navigation_plan, get_navigation_actions
from thortils.utils import (to_degrees, closest,
                            normalize_angles, euclidean_dist)
from .utils import plot_path, plt
from .task import TOS
from ..framework import Agent

class ThorAgent(Agent):
    def __init__(self, movement_params):
        self.movement_params = movement_params
        # The navigation_actions here is a list of tuples
        # (movement_str, (forward, h_angle, v_angle))
        self.navigation_actions =\
            get_navigation_actions(self.movement_params)

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

    def __init__(self, controller, task_config,
                 movement_params):
        super().__init__(movement_params)
        self.controller = controller
        self.target = task_config["target"]
        self.task_type = task_config["task_type"]
        self.goal_distance = task_config["goal_distance"]
        # The valid rotation angles horizontally and vertically
        # https://github.com/allenai/ai2thor/blob/68edec39b5f94bbc6532aaac5ed4ee50f4b09bb1/ai2thor/controller.py#L1282
        self.v_angles = task_config["v_angles"]
        self.h_angles = task_config["h_angles"]

        if self.task_type == "class":
            obj = thor_closest_object_of_type(controller, self.target)
            target_position = (obj["position"]["x"],
                               obj["position"]["y"],
                               obj["position"]["z"])
        else:
            target_position = thor_object_pose(controller,
                                               self.target, as_tuple=True)

        start_pose = thor_agent_pose(self.controller, as_tuple=True)
        start_position, start_rotation = start_pose

        reachable_positions = thor_reachable_positions(controller)
        goal_pose = self._goal_pose(reachable_positions,
                                    target_position,
                                    y=start_position[1],
                                    roll=start_rotation[2])

        plan = find_navigation_plan(start_pose, goal_pose,
                                    self.navigation_actions,
                                    reachable_positions,
                                    goal_distance=self.goal_distance)
        self.plan = plan
        self._index = 0

    def act(self):
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

    def _yaw_facing(self, robot_position, target_position, angles):
        """
        Returns a yaw angle rotation such that
        if the robot is at `robot_position` and target is at
        `target_position`, the robot is facing the target.

        Args:
           robot_position (tuple): x, y, z position
           target_position (tuple): x, y, z position
           angles (list): Valid yaw angles
        """
        angles = normalize_angles(angles)
        rx, _, rz = robot_position
        tx, _, tz = target_position
        yaw = to_degrees(math.atan2(tx - rx, tz - rz)) % 360
        return closest(angles, yaw)

    def _pitch_facing(self, robot_position, target_position, angles):
        """
        Returns a pitch angle rotation such that
        if the robot is at `robot_position` and target is at
        `target_position`, the robot is facing the target.

        Args:
           robot_position (tuple): x, y, z position
           target_position (tuple): x, y, z position
           angles (list): Valid pitch angles
        """
        angles = normalize_angles(angles)
        rx, ry, _ = robot_position
        tx, ty, _ = target_position
        pitch = to_degrees(math.atan2(ty - ry, tx - rx)) % 360
        return closest(angles, pitch)

    def _goal_pose(self, reachable_positions,
                   target_position, y=0.0, roll=0.0):
        """Compute the goal pose; the rotation should be facing the target"""
        x, z =\
            min(reachable_positions,
                key=lambda p : euclidean_dist(p, target_position))
        closest_reachable_pos = (x, y, z)
        goal_pitch = self._pitch_facing(closest_reachable_pos,
                                        target_position, self.v_angles)
        goal_yaw = self._yaw_facing(closest_reachable_pos,
                                    target_position, self.h_angles)
        goal_pose = (target_position, (goal_pitch, goal_yaw, roll))
        return goal_pose
