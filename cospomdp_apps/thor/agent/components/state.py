from cospomdp.domain.state import RobotState, RobotStatus, RobotState2D

class RobotStateTopo(RobotState):
    def __init__(self, robot_id, pose, horizon, topo_nid, status=RobotStatus()):
        """
        We treat robot pose in the same way as Ai2thor does:
           pose (x, y, yaw): The position and rotation of the base
           horizon (float): The pitch of the camera (tilt up and down)
        """
        super().__init__(robot_id, pose, status)
        self.topo_nid = topo_nid
        self.horizon = horizon        #used?

    @property
    def pitch(self):
        #used?
        return self.horizon

    @property
    def nid(self):
        return self.topo_nid

    @property
    def loc(self):
        return self['pose'][:2]

    @staticmethod
    def from_obz(robot_obz):
        """
        robot_obz (RobotObservation); Here, we will receive pose
            as (x, y, pitch, yaw, nid) in the robot_obz. The fields
            of RobotObservation are not changed.
        """
        x, y, yaw = robot_obz.pose
        return RobotStateTopo(robot_obz.robot_id,
                              (x, y, yaw),
                              robot_obz.horizon,
                              robot_obz.topo_nid,
                              robot_obz.status)

class RobotState3D(RobotState):
    """
     We treat robot pose in the same way as Ai2thor does:
        pose (x, y, yaw): The position and rotation of the base
        horizon (float): The pitch of the camera (tilt up and down)
    """
    def __init__(self, robot_id, pose,
                 camera_height, camera_horizon, status=RobotStatus()):
        super().__init__(robot_id, pose, status)
        self.horizon = camera_horizon
        self.height = camera_height  # the robot's own height

    @property
    def pitch(self):
        return self.horizon

    @property
    def camera_height(self):
        return self.height

    @property
    def loc(self):
        return self['pose'][:2]

    @staticmethod
    def from_obz(robot_obz):
        """
        robot_obz (RobotObservation); Here, we will receive pose
            as (x, y, pitch, yaw, nid) in the robot_obz. The fields
            of RobotObservation are not changed.
        """
        x, y, yaw = robot_obz.pose
        return RobotState3D(robot_obz.robot_id,
                            (x, y, yaw),
                            robot_obz.camera_height,
                            robot_obz.horizon,
                            robot_obz.status)

    @property
    def pose3d(self):
        x, y, yaw = self.pose
        return (x, y, self.pitch, yaw)

    def to_2d(self):
        return RobotState2D(self.robot_id, self.pose, self.status)

    @property
    def loc3d(self):
        return (*self.loc, self.height)
