from pomdp_py import TransitionModel
from thortils.navigation import transform_pose, _is_full_pose

from ..utils.math import indicator
from .action import Move
from .state import ObjectState2D

def robot_pose_transition(robot_pose, action,
                          grid_size=None, diagonal_ok=False):
    """
    Uses the transform_pose function to compute the next pose,
    given a robot pose and an action.

    Args:
        robot_pose (position, rotation), or (x, y, th)
        action (MoveAction)
        see transform_pose for kwargs [grid_size, diagonal_ok]
    """
    if _is_full_pose(robot_pose):
        return transform_pose(robot_pose, (action.name, action.delta),
                              grid_size=grid_size, diagonal_ok=diagonal_ok, schema="vw")

    elif len(robot_pose) == 3:
        return transform_pose(robot_pose, (action.name, action.delta),
                              grid_size=grid_size, diagonal_ok=diagonal_ok, schema="vw2d")
    else:
        raise ValueError("Unrecognized robot_pose format {}".format(robot_pose))


class RobotTransition2D(TransitionModel):
    def __init__(self, reachable_positions, grid_size=None, diagonal_ok=False):
        """Snap to grid if `grid_size` is None. Otherwise, continuous."""
        self.reachable_positions = reachable_positions
        self._grid_size = grid_size
        self._diagonal_ok = diagonal_ok

    def sample(self, state, action):
        current_robot_pose = state.robot_state["pose"]
        next_robot_pose = current_robot_pose
        if isinstance(action, Move):
            next_robot_pose = robot_pose_transition(
                current_robot_pose, action, grid_size=self._grid_size,
                diagonal_ok=self._diagonal_ok)
        if next_robot_pose[:2] not in self.reachable_positions:
            return ObjectState2D("robot", dict(pose=current_robot_pose))
        else:
            return ObjectState2D("robot", dict(pose=next_robot_pose))
