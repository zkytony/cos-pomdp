import math
import numpy as np
import pomdp_py
from pomdp_py import RewardModel
from ..utils.math import euclidean_dist, to_rad
from ..domain.action import Done

class ObjectSearchRewardModel2D(pomdp_py.RewardModel):
    def __init__(self, sensor, goal_dist, robot_id, target_id,
                 hi=100, lo=-100, step=-1):
        self.sensor = sensor
        self.goal_dist = goal_dist
        self.robot_id = robot_id
        self.target_id = target_id
        self._hi = hi
        self._lo = lo
        self._step = step

    def sample(self, state, action, next_state):
        srobot = state.s(self.robot_id)
        if srobot.done:
            return 0  # the robot is already done.
        robot_pose = srobot["pose"]
        target_loc = next_state.s(self.target_id)["loc"]
        if isinstance(action, Done):
            if self.success2d(robot_pose, target_loc):
                return self._hi
            else:
                return self._lo
        return self._step

    def success2d(self, robot_pose, target_loc):
        x, y, th = robot_pose
        if euclidean_dist((x,y), target_loc) <= self.goal_dist:
            if self.sensor.in_range(target_loc, robot_pose):
                return True
        return False

class NavRewardModel2D(pomdp_py.RewardModel):
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
        robot_pose = next_state.s(self.robot_id)["pose"]
        rx, ry, rth = robot_pose
        gx, gy, gth = self.goal_pose
        if isinstance(action, Done):
            if (rx, ry) == (gx, gy)\
               and (rth % 360 - gth % 360) < 15:
                return self._hi
            else:
                return self._lo
        return self._step
