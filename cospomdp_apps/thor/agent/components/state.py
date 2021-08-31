from cospomdp.domain.state import RobotState, RobotStatus

class RobotStateTopo(RobotState):
    def __init__(self, robot_id, pose, horizon, topo_nid, status=RobotStatus()):
        """
        We treat robot pose in the same way as Ai2thor does:
           pose (x, y, yaw): The position and rotation of the base
           horizon (float): The pitch of the camera (tilt up and down)
        """
        super().__init__(robot_id, pose, status)
        self.topo_nid = topo_nid
        self.horizon = horizon

    @property
    def pitch(self):
        return self.horizon

    @property
    def nid(self):
        return self.topo_nid

    @property
    def loc(self):
        return self['pose'][:2]
