import math
from pomdp_py import TransitionModel

from ..utils.math import indicator, to_rad
from .action import Move
from .state import ObjectState2D, JointState2D

def robot_pose_transition2d(robot_pose, action, diagonal_ok=False):
    """
    Uses the transform_pose function to compute the next pose,
    given a robot pose and an action.

    Note: robot_pose is a 2D POMDP (gridmap) pose.

    Args:
        robot_pose (x, y, th)
        action (MoveAction)
        see transform_pose for kwargs [grid_size, diagonal_ok]
    """
    rx, ry, rth = robot_pose
    forward, angle = action.delta
    nth = rth + angle
    nx = rx + forward*math.cos(to_rad(nth))
    ny = ry + forward*math.sin(to_rad(nth))
    if diagonal_ok:
        next_robot_pose = (int(round(nx)), int(round(ny)), nth)
    else:
        next_robot_pose = (int(nx), int(ny), nth)
    return next_robot_pose

class RobotTransition2D(TransitionModel):
    def __init__(self, robot_id, reachable_positions, diagonal_ok=False):
        """Snap to grid if `grid_size` is None. Otherwise, continuous."""
        self.robot_id = robot_id
        self.reachable_positions = reachable_positions
        self._diagonal_ok = diagonal_ok

    def sample(self, state, action):
        current_robot_pose = state.robot_state["pose"]
        next_robot_pose = current_robot_pose
        if isinstance(action, Move):
            next_robot_pose = robot_pose_transition2d(
                current_robot_pose, action, diagonal_ok=self._diagonal_ok)
        if next_robot_pose[:2] not in self.reachable_positions:
            return ObjectState2D(self.robot_id, dict(pose=current_robot_pose))
        else:
            return ObjectState2D(self.robot_id,  dict(pose=next_robot_pose))

class JointTransitionModel2D(TransitionModel):
    def __init__(self, target_id, robot_trans_model):
        self.target_id = target_id
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        starget = state.target_state
        next_target_state = ObjectState2D(starget.objclass,
                                          dict(loc=starget["loc"]))
        robot_id = self.robot_trans_model.robot_id
        return JointState2D(robot_id,
                            self.target_id,
                            {robot_id: next_robot_state,
                             self.target_id: next_target_state})
