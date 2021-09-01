import math
from cospomdp.utils.math import to_deg, closest
from cospomdp.models.transition_model import RobotTransition
from .action import MoveTopo
from .state import RobotStateTopo

def _yaw_facing(robot_pos, target_pos, angles):
    rx, ry = robot_pos
    tx, ty = target_pos
    yaw = to_deg(math.atan2(ty - ry,
                            tx - rx)) % 360
    return closest(angles, yaw)

class RobotTransitionTopo(RobotTransition):

    def __init__(self, robot_id, target_id, topo_map, h_angles):
        super().__init__(robot_id)
        self._topo_map = topo_map
        self.target_id = target_id
        self.h_angles = h_angles

    def sample(self, state, action):
        srobot = state.s(self.robot_id)
        starget = state.s(self.target_id)

        if isinstance(action, MoveTopo):
            if srobot.nid == action.src_nid:
                next_robot_pos = self._topo_map.nodes[action.dst_nid].pos

                # will sample a yaw facing the target object
                yaw = _yaw_facing(next_robot_pos, starget.loc, self.h_angles)

                # there is no pitch
                pitch = 0.0

                next_srobot = RobotStateTopo(srobot.id,
                                             (*next_robot_pos, yaw),
                                             pitch,
                                             action.dst_nid)
            else:
                raise ValueError("Unexpected action {} for robot state {}.".format(action, srobot))
        else:
            next_srobot = RobotStateTopo(srobot.id,
                                         srobot.pose,
                                         srobot.horizon,
                                         srobot.nid)

        return next_srobot

    def update(self, topo_map):
        self._topo_map = topo_map
