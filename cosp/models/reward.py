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

def _facing2d(robot_pose, point):
    rx, ry, th = robot_pose
    if (rx, ry) == point:
        return True
    # point in direction of robot facing
    rx2 = rx + math.sin(to_rad(th))
    ry2 = ry + math.cos(to_rad(th))
    px, py = point
    return np.dot(np.array([px - rx, py - ry]),
                  np.array([rx2 - rx, ry2 - ry])) > 0

def thor_success2d(robot_pose, target_loc,
                   dist_thresh=GOAL_DISTANCE):
    x, y, th = robot_pose
    if euclidean_dist((x,y), target_loc)*GRID_SIZE <= dist_thresh:
        if _facing2d(robot_pose, target_loc):
            return True
    return False

class ThorRewardModel2D(RewardModel):
    def sample(self, state, action, next_state):
        robot_pose = next_state.robot_state["pose"]
        target_loc = next_state.target_state["loc"]
        if isinstance(action, Done):
            if thor_success2d(robot_pose, target_loc):
                return TOS_REWARD_HI
            else:
                return TOS_REWARD_LO
        return TOS_REWARD_STEP
