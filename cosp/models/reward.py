import math
import numpy as np
from pomdp_py import RewardModel
from ..thor.constants import (GOAL_DISTANCE,
                              TOS_REWARD_HI,
                              TOS_REWARD_LO,
                              TOS_REWARD_STEP,
                              GRID_SIZE)
from ..utils.math import euclidean_dist, to_rad
from .action import Done

class ObjectSearchRewardModel2D(RewardModel):
    def __init__(self, sensor, hi=100, lo=-100, step=-1):
        self.sensor = sensor
        self._hi = hi
        self._lo = lo
        self._step = step

    def sample(self, state, action, next_state):
        robot_pose = next_state.robot_state["pose"]
        target_loc = next_state.target_state["loc"]
        if isinstance(action, Done):
            if self.success2d(robot_pose, target_loc):
                return self._hi
            else:
                return self._lo
        return self._step

    def success2d(self, robot_pose, target_loc,
                  dist_thresh=GOAL_DISTANCE):
        x, y, th = robot_pose
        if euclidean_dist((x,y), target_loc)*GRID_SIZE <= dist_thresh:
            # robot_pose = (robot_pose[0], robot_pose[1], (robot_pose[2]-135)%360.0)
            if self.sensor.in_range(target_loc, robot_pose):
                return True
        return False
