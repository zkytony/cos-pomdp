from cospomdp.domain.state import RobotState, RobotStatus, RobotState2D, ObjectState
from cospomdp.utils.math import indicator, normalize, euclidean_dist, roundany, closest
from cospomdp.models.sensors import FanSensor3D
from .action import grid_pitch


def grid_full_pose(thor_pose, thor_v_angles, grid_map):
    thor_pos, thor_rot = thor_pose
    x, y, yaw = grid_map.to_grid_pose(
        thor_pos['x'],  #x
        thor_pos['z'],  #z
        thor_rot['y']   #yaw
    )

    pitch = grid_pitch(
        closest(thor_v_angles, thor_rot['x']))
    height = roundany(thor_pos['y'] / grid_map.grid_size, 1)  #y
    return (x, y, height, pitch, yaw)



class ObjectState3D(ObjectState):
    def __init__(self, objid, objclass, loc, height):
        super().__init__(objid, objclass, loc)
        self.attributes["height"] = height

    def __hash__(self):
        return hash((self.id, self.loc, self.height))

    @property
    def height(self):
        return self["height"]

    def to_2d(self):
        return ObjectState(self.id, self.objclass, self['loc'])

    @property
    def loc3d(self):
        return (*self.loc, self.height)

    def __str__(self):
        return str((self.id, *self.loc, self.height))

    def copy(self):
        return ObjectState3D(self.id,
                             self.objclass,
                             self.loc,
                             self.height)


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
        self.height = camera_height  # the robot's own height, should be fixed

    @property
    def pitch(self):
        return self.horizon

    @property
    def robot_id(self):
        return self.id

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
                            robot_obz.height,
                            robot_obz.horizon,
                            robot_obz.status)

    @property
    def pose3d(self):
        x, y, yaw = self.pose
        return (x, y, self.height, self.pitch, yaw)

    def to_2d(self):
        return RobotState2D(self.robot_id, self.pose, self.status)

    @property
    def loc3d(self):
        return (*self.loc, self.height)

    def __str__(self):
        return str((self.robot_id, *self.pose3d))

    def in_range(self, sensor, loc, **kwargs):
        if isinstance(sensor, FanSensor3D):
            return sensor.in_range(loc, self.pose3d, **kwargs)
        else:
            return sensor.in_range(loc, self.pose, **kwargs)

    def in_range_facing(self, sensor, point, **kwargs):
        if isinstance(sensor, FanSensor3D):
            return sensor.in_range_facing(point, self.pose3d, **kwargs)
        else:
            return sensor.in_range_facing(point, self.pose, **kwargs)

class RobotStateTopo(RobotState3D):
    def __init__(self, robot_id, pose,
                 camera_height, camera_horizon,
                 topo_nid, status=RobotStatus()):
        """
        We treat robot pose in the same way as Ai2thor does:
           pose (x, y, yaw): The position and rotation of the base
           horizon (float): The pitch of the camera (tilt up and down)
        """
        super().__init__(robot_id, pose, camera_height, camera_horizon, status)
        self.topo_nid = topo_nid

    @property
    def nid(self):
        return self.topo_nid

    def __str__(self):
        return f"({self.robot_id}, {self.pose3d}, topo_nid({self.topo_nid})"

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
                              robot_obz.height,
                              robot_obz.horizon,
                              robot_obz.topo_nid,
                              robot_obz.status)
