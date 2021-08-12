from pomdp_py.utils import typ
from cosp.models.state import *
from cosp.models.action import *
from cosp.models.observation import (ObjectDetection2D,
                                     ObjectDetection3D)

def test_state_creation():
    robot_state = ObjectState2D("robot", {"pose": ((0, 1, 0), (0, 270, 0))})
    object_state = ObjectState2D("object", {"loc": (0, 5)})
    s1 = JointState2D({1:robot_state, 2:object_state})
    s2 = JointState2D({1:robot_state, 2:object_state})
    hash(object_state)
    hash(robot_state)
    hash(s1)
    hash(s2)
    assert s1 == s2

def test_observation_creation():
    z1 = ObjectDetection2D("mug", (5,5))
    z2 = ObjectDetection2D("glass", (5,5))
    hash(z1)
    hash(z2)
    assert z1 != z2
    z1 = ObjectDetection3D("mug", (5,5,7))
    z2 = ObjectDetection3D("glass", (5,5,7))
    hash(z1)
    hash(z2)
    assert z1 != z2
    try:
        ObjectDetection3D("mug", (5,5))
    except AssertionError:
        pass
    try:
        ObjectDetection2D("mug", (5,5,7))
    except AssertionError:
        pass

def test_action_creation():
    a1 = Move("north", (-1, 1))
    a2 = Move("north", (-1, 1))
    assert(a1 == a2)
    hash(a1)
    hash(a2)
    a3 = Interact("open", "drawer", {"openness": 90})
    hash(a3)
    print(typ.info(str(a3)))


if __name__ == "__main__":
    test_state_creation()
    test_observation_creation()
    test_action_creation()
    print("Passed.")
