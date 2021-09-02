import math
from cospomdp.utils.math import to_deg, closest
from cospomdp.models.transition_model import RobotTransition
from cospomdp.models.sensors import yaw_facing
from cospomdp.domain.action import Done
from cospomdp.domain.state import RobotStatus
from .action import MoveTopo
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
                                     next_horizon,
                                     next_topo_nid,
                                     next_status)

        return next_srobot

    def update(self, topo_map):
        self._topo_map = topo_map
