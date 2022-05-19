# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# All locations are in GridMap coordinate system.

import pomdp_py
from dataclasses import dataclass  #https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple/34269877

class ObjectState(pomdp_py.ObjectState):
    """Object state, specified by object ID, class and its location"""
    def __init__(self, objid, objclass, loc):
        super().__init__(objclass, {"loc": loc, "id": objid})

    def __hash__(self):
        return hash((self.id, self.loc))

    @property
    def loc(self):
        return self['loc']

    @property
    def id(self):
        return self['id']

    def __lt__(self, other):
        if not isinstance(other, ObjectState):
            raise ValueError("Cannot compare ObjectState with {}".format(other.__class__.__name__))
        return self.id < other.id\
            and self.loc < other.loc

    def copy(self):
        return ObjectState(self.id,
                           self.objclass,
                           self.loc)

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

    def __hash__(self):
        return hash(self.pose)

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

    @staticmethod
    def from_obz(robot_obz):
        """
        robot_obz (RobotObservation)
        """
        raise NotImplementedError

    def in_range(self, sensor, loc):
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

    @staticmethod
    def from_obz(robot_obz):
        """
        robot_obz (RobotObservation)
        """
        return RobotState2D(robot_obz.robot_id,
                            robot_obz.pose,
                            robot_obz.status)

    def in_range(self, sensor, sobj, **kwargs):
        return sensor.in_range(sobj.loc, self["pose"], **kwargs)

    def loc_in_range(self, sensor, loc, **kwargs):
        return sensor.in_range(loc, self["pose"], **kwargs)

    def in_range_facing(self, sensor, sobj, **kwargs):
        return sensor.in_range_facing(sobj.loc, self["pose"], **kwargs)


class CosState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)

    def s(self, objid):
        return self.object_states[objid]

    def __str__(self):
        return "{}({})".format(self.__class__, self.object_states)

    def __repr__(self):
        return str(self)
