from pomdp_py import OOState, ObjectState

class JointState(OOState):
    def __init__(self, robot_id, target_id, object_states):
        self.robot_id = robot_id
        self.target_id = target_id
        super().__init__(object_states)
    @property
    def robot_state(self):
        return self.object_states[self.robot_id]
    @property
    def target_state(self):
        return self.object_states[self.target_id]

class JointState2D(JointState):
    pass

class JointState3D(JointState):
    pass

class ObjectState2D(ObjectState):
    pass

class ObjectState3D(ObjectState):
    pass

class RobotState(ObjectState3D):
    def __init__(self, pose):
        assert len(pose) == 7,\
            "Robot pose needs to be position, rotation"\
             "where rotation is represented in quaternion"
        super().__init__("robot", dict(pose=pose))

class PhysicalObjectState(ObjectState3D):
    def __init__(self, objclass, loc, coords=None):
        """
        loc (x,y,z) 3D object location
        coords (list of 3D coordinates) specifies coordinates
            that the object occupies, when the object is placed at
            loc=(0,0,0)
        """
        self.coords = coords
        if coords is None:
            self.coords = [(0,0,0)]
        super().__init__(objclass, dict(loc=loc))

    def space_occupying(self):
        x, y, z = self["loc"]
        return [(x + c[0],
                 y + c[1],
                 z + c[2])
                for c in self.coords]
