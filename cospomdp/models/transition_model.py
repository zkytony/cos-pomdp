import math
from pomdp_py import TransitionModel

from ..utils.math import indicator, to_rad, fround
from ..domain.action import Move2D, Done
from ..domain.state import RobotState2D, CosState2D, ObjectState2D, RobotStatus

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

class RobotTransition2D(TransitionModel):
    def __init__(self, robot_id, reachable_positions, round_to='int'):
        """round_to: round the x, y coordinates to integer, floor integer,
        or not rounding, when computing the next robot pose."""
        self.robot_id = robot_id
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

class CosTransitionModel2D(TransitionModel):
    def __init__(self, target_id, robot_trans_model):
        self.target_id = target_id
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        starget = state.s(self.target_id)
        next_target_state = ObjectState2D(starget.id, starget.objclass, starget["loc"])
        robot_id = self.robot_trans_model.robot_id
        return CosState2D({robot_id: next_robot_state,
                           self.target_id: next_target_state})


class FullTransitionModel2D(TransitionModel):
    def __init__(self, robot_trans_model):
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        objstates = {next_robot_state.id:next_robot_state}
        for objid in state.object_states:
            if objid == next_robot_state.id:
                continue
            next_object_state = ObjectState2D(objid,
                                              state.s(objid).objclass,
                                              state.s(objid)['loc'])
            objstates[objid] = next_object_state
        return CosState2D(objstates)
