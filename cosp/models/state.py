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
