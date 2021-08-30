# All locations are in GridMap coordinate system.

import pomdp_py
from dataclasses import dataclass  #https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple/34269877

class ObjectState(pomdp_py.ObjectState):
    """Object state, specified by object ID, class and its location"""
    def __init__(self, objid, objclass, loc):
        super().__init__(objclass, {"loc": loc, "id": objid})

    @property
    def loc(self):
        return self['loc']

    @property
    def id(self):
        return self['id']

@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class RobotStatus:
    # One feature of dataclass frozen is the fields cannot be reassigned
    done: bool = False
    def __str__(self):
        done_status = "done" if self.done else "in prorgess"
        return done_status

    def copy(self):
        return RobotStatus(self.done)

class RobotState(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, status=RobotStatus()):
        super().__init__("robot",
                         {"id": robot_id,
                          "pose": pose,
                          "status": status})
    def __str__(self):
        return "{}({}, {})".format(self.__class__, self.pose, self.status)

    @property
    def pose(self):
        return self["pose"]

    @property
    def status(self):
        return self["status"]

    @property
    def done(self):
        return self.status.done

    @property
    def id(self):
        return self['id']

    @property
    def loc(self):
        """the location of the robot, regardless of orientation"""
        raise NotImplementedError


class RobotState2D(RobotState):
    """2D robot state; pose is x, y, th"""
    @property
    def loc(self):
        return self['pose'][:2]

    def same_pose(self, other_pose):
        rx, ry, rth = self['pose']
        gx, gy, gth = other_pose
        if (rx, ry) == (gx, gy):
            if abs(rth % 360 - gth % 360) <= 15:
                return True
        return False

class CosState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)

    def s(self, objid):
        return self.object_states[objid]

    def __str__(self):
        return "{}({})".format(self.__class__, self.object_states)

    def __repr__(self):
        return str(self)
