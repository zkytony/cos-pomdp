import math
import numpy as np
import pomdp_py
from pomdp_py import RewardModel
from ..utils.math import euclidean_dist, to_rad
from ..domain.action import Done

class ObjectSearchRewardModel(pomdp_py.RewardModel):
    def __init__(self,
                 sensor, goal_dist, robot_id, target_id,
                 hi=100, lo=-100, step=-1):
        self.sensor = sensor
        self.goal_dist = goal_dist
        self.robot_id = robot_id
        self.target_id = target_id
        self._hi = hi
        self._lo = lo
        self._step = step  # default step cost

    def sample(self, state, action, next_state):
        srobot = state.s(self.robot_id)
        if srobot.done:
            return 0  # the robot is already done.
        starget = next_state.s(self.target_id)
        if isinstance(action, Done):
            if self.success(srobot, starget):
                return self._hi
            else:
                return self._lo
        if hasattr(action, "step_cost"):
            return action.step_cost
        else:
            return self._step

    def success(self, srobot, starget):
        if euclidean_dist(srobot.loc, starget.loc) <= self.goal_dist:
            if self.sensor.in_range(starget.loc, srobot.pose):
                return True
        return False


class NavRewardModel(pomdp_py.RewardModel):
    def __init__(self, goal_pose, robot_id,
                 hi=100, lo=-100, step=-1):
        self.goal_pose = goal_pose
        self.robot_id = robot_id
        self._hi = hi
        self._lo = lo
        self._step = step

    def sample(self, state, action, next_state):
        srobot = state.s(self.robot_id)
        if srobot.done:
            return 0  # the robot is already done.
        next_srobot = next_state.s(self.robot_id)
        if isinstance(action, Done):
            if next_srobot.same_pose(self.goal_pose):
                return self._hi
            else:
                return self._lo
        if hasattr(action, "step_cost"):
            return action.step_cost
        else:
            return self._step
