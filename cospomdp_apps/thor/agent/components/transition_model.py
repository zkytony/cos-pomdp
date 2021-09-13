import math
from cospomdp.utils.math import to_deg, closest, to_rad, fround
from cospomdp.models.transition_model import RobotTransition
from cospomdp.models.sensors import yaw_facing
from cospomdp.domain.action import Done
from cospomdp.domain.state import RobotStatus
from .state import RobotState3D
from .action import MoveTopo, Move
from .state import RobotStateTopo

class RobotTransitionTopo(RobotTransition):

    def __init__(self, robot_id, target_id, topo_map, h_angles):
        super().__init__(robot_id)
        self._topo_map = topo_map
        self.target_id = target_id
        self.h_angles = h_angles

    def sample(self, state, action):
        srobot = state.s(self.robot_id)
        starget = state.s(self.target_id)

        next_pose = srobot["pose"]
        next_status = srobot.status.copy()
        next_height = srobot.height
        next_horizon = srobot.horizon
        next_topo_nid = srobot.nid

        if isinstance(action, MoveTopo):
            if srobot.nid == action.src_nid:
                next_robot_pos = self._topo_map.nodes[action.dst_nid].pos
                # will sample a yaw facing the target object
                yaw = yaw_facing(next_robot_pos, starget.loc, self.h_angles)
                next_pose = (*next_robot_pos, yaw)
                next_horizon = 0.0 # there is no pitch
                next_topo_nid = action.dst_nid
            else:
                raise ValueError("Unexpected action {} for robot state {}.".format(action, srobot))
        elif isinstance(action, Done):
            next_status = RobotStatus(done=True)

        next_srobot = RobotStateTopo(srobot.id,
                                     next_pose,
                                     next_height,
                                     next_horizon,
                                     next_topo_nid,
                                     next_status)

        return next_srobot

    def update(self, topo_map):
        self._topo_map = topo_map


def robot_pose_transition3d(robot_pose, action):
    """
    Uses the transform_pose function to compute the next pose,
    given a robot pose and an action.

    Note: robot_pose is a 2D POMDP (gridmap) pose.

    Args:
        robot_pose (x, y, th)
        action (Move2D)
    """
    rx, ry, pitch, yaw = robot_pose
    forward, h_angle, v_angle = action.delta
    new_yaw = (yaw + h_angle) % 360
    nx = rx + forward*math.cos(to_rad(new_yaw))
    ny = ry + forward*math.sin(to_rad(new_yaw))
    new_pitch = (pitch + v_angle) % 360
    return (nx, ny, new_pitch, new_yaw)

def _to_full_pose(srobot):
    x, y, yaw = srobot["pose"]
    pitch = srobot["horizon"]
    z = srobot["height"]
    return (x, y, z, pitch, yaw)

def _to_state_pose(full_pose):
    x, y, z, pitch, yaw = full_pose
    return (x, y, yaw), z, pitch


class RobotTransition3D(RobotTransition):
    def __init__(self, robot_id, reachable_positions, v_angles, round_to="int"):
        super().__init__(robot_id)
        self.reachable_positions = reachable_positions
        self._round_to = round_to
        self._v_angles = v_angles

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        srobot = state.s(self.robot_id)
        current_robot_pose = _to_full_pose(srobot["pose"])
        next_robot_pose = current_robot_pose
        next_robot_status = srobot.status.copy()
        if isinstance(action, Move):
            np = robot_pose_transition3d(current_robot_pose, action)
            next_robot_pose = fround(self._round_to, np)
        elif isinstance(action, Done):
            next_robot_status = RobotStatus(done=True)

        next_pose2d, height, pitch = _to_state_pose(current_robot_pose)
        if pitch not in self._v_angles\
           or next_pose2d[:2] not in self.reachable_positions:
            return RobotState3D(self.robot_id, srobot["pose"],
                                height, srobot.horizon, next_robot_status)
        else:
            return RobotState3D(self.robot_id, next_pose2d,
                                height, pitch, next_robot_status)
