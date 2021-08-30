# This instantiates the robot transition model used in the COS-POMDP
# created for this domain.
import math
import random
from pomdp_py import TransitionModel
from cospomdp.domain.action import Done
from cospomdp.domain.state import RobotState2D, RobotStatus
from cospomdp.models.transition_model import RobotTransition
from cospomdp.utils.math import indicator, to_rad, fround, euclidean_dist
from .action import Move2D

def robot_pose_transition2d(robot_pose, action):
    """
    Uses the transform_pose function to compute the next pose,
    given a robot pose and an action.

    Note: robot_pose is a 2D POMDP (gridmap) pose.

    Args:
        robot_pose (x, y, th)
        action (Move2D)
    """
    rx, ry, rth = robot_pose
    forward, angle = action.delta
    nth = (rth + angle) % 360
    nx = rx + forward*math.cos(to_rad(nth))
    ny = ry + forward*math.sin(to_rad(nth))
    return (nx, ny, nth)

############################
# Robot Transition
############################
class RobotTransition2D(RobotTransition):
    def __init__(self, robot_id, reachable_positions, round_to='int'):
        """round_to: round the x, y coordinates to integer, floor integer,
        or not rounding, when computing the next robot pose."""
        super().__init__(robot_id)
        self.reachable_positions = reachable_positions
        self._round_to = round_to

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        srobot = state.s(self.robot_id)
        current_robot_pose = srobot["pose"]
        next_robot_pose = current_robot_pose
        next_robot_status = srobot.status.copy()
        if isinstance(action, Move2D):
            np = robot_pose_transition2d(current_robot_pose, action)
            next_robot_pose = fround(self._round_to, np)
        elif isinstance(action, Done):
            next_robot_status = RobotStatus(done=True)

        if next_robot_pose[:2] not in self.reachable_positions:
            return RobotState2D(self.robot_id, current_robot_pose, next_robot_status)
        else:
            return RobotState2D(self.robot_id, next_robot_pose, next_robot_status)
