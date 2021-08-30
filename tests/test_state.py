from cospomdp.domain.state import *

def test_state_creation():
    robot_state = RobotState(0, (0, 1, 90), RobotStatus(False))
    assert robot_state.pose[2] == 90
    assert robot_state.status.done == robot_state.done

    object_state = ObjectState(1, "vase", (5,5))
    assert object_state.id == 1
    assert object_state.loc == (5,5)

    joint_state = CosState({0:robot_state, 1:object_state})
    assert joint_state.s(0) == robot_state
    assert joint_state.s(1) == object_state
