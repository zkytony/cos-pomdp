import pytest
from cospomdp.models.reward_model import ObjectSearchRewardModel2D, NavRewardModel2D
from cospomdp.models.transition_model import RobotTransition2D, CosTransitionModel2D
from cospomdp.models.sensors import FanSensor
from cospomdp.models.search_region import SearchRegion2D
from cospomdp.domain.state import RobotState2D, ObjectState2D, CosState2D
from cospomdp.domain.action import MoveAhead, RotateLeft, Done

@pytest.fixture
def fansensor():
    return FanSensor(fov=75, min_range=0, max_range=4)

@pytest.fixture
def init_srobot():
    init_pose = (2, 5, 0)
    return RobotState2D("robot", init_pose)

@pytest.fixture
def search_region():
    w, l = 15, 15
    locations = [(x,y) for x in range(w) for y in range(l)]
    search_region = SearchRegion2D(locations)
    return search_region

def test_reward_model_object_search(fansensor, init_srobot, search_region):
    robot_id = 0
    target_id = 10

    Trobot = RobotTransition2D(robot_id, search_region.locations)
    T = CosTransitionModel2D(target_id, Trobot)
    R = ObjectSearchRewardModel2D(fansensor, 2.0, robot_id, target_id)

    starget = ObjectState2D(target_id, "target", (4, 5))
    state = CosState2D({robot_id: init_srobot,
                        target_id: starget})
    assert R.sample(state, Done(), T.sample(state, Done())) == R._hi
    assert R.sample(state, MoveAhead, T.sample(state, MoveAhead)) == R._step

    state2 = T.sample(T.sample(state, RotateLeft), RotateLeft)
    assert R.sample(state2, Done(), T.sample(state2, Done())) == R._lo


def test_reward_model_nav(init_srobot, search_region):
    robot_id = 0
    target_id = 10

    Trobot = RobotTransition2D(robot_id, search_region.locations)
    T = CosTransitionModel2D(target_id, Trobot)
    R = NavRewardModel2D((2, 6, 135), robot_id)
    starget = ObjectState2D(target_id, "target", (4, 5))
    state = CosState2D({robot_id: init_srobot,
                        target_id: starget})

    R.sample(state, Done(), T.sample(state, Done())) == R._lo

    state2 = T.sample(T.sample(T.sample(T.sample(state, RotateLeft), RotateLeft), MoveAhead), RotateLeft)
    R.sample(state2, Done(), T.sample(state2, Done())) == R._lo
    R.sample(state2, MoveAhead, T.sample(state, MoveAhead)) == R._step
