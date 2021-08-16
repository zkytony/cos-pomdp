import pytest

from cospomdp.models.policy_model import PolicyModel2D
from cospomdp.models.reward_model import ObjectSearchRewardModel2D, NavRewardModel2D
from cospomdp.models.transition_model import RobotTransition2D, CosTransitionModel2D
from cospomdp.models.sensors import FanSensor
from cospomdp.models.search_region import SearchRegion2D
from cospomdp.domain.state import RobotState2D, ObjectState2D, CosState2D
from cospomdp.domain.action import MoveAhead, RotateLeft, Done, ALL_MOVES_2D

@pytest.fixture
def robot_id():
    return 0

@pytest.fixture
def target_id():
    return 100

@pytest.fixture
def fansensor():
    return FanSensor(fov=75, min_range=0, max_range=4)

@pytest.fixture
def init_srobot(robot_id):
    init_pose = (2, 5, 0)
    return RobotState2D(robot_id, init_pose)

@pytest.fixture
def search_region():
    w, l = 15, 15
    locations = [(x,y) for x in range(w) for y in range(l)]
    search_region = SearchRegion2D(locations)
    return search_region

@pytest.fixture
def robot_trans_model(robot_id, search_region):
    return RobotTransition2D(robot_id, search_region.locations)

@pytest.fixture
def objsearch_reward_model(fansensor, robot_id, target_id):
    # 2.0 is the goal distance
    return ObjectSearchRewardModel2D(fansensor, 2.0, robot_id, target_id)

@pytest.fixture
def nav_reward_model(robot_id):
    goal = (2, 6, 135)
    return NavRewardModel2D(goal, robot_id)

def test_policy_model_object_search(robot_id,
                                    target_id,
                                    init_srobot,
                                    robot_trans_model,
                                    objsearch_reward_model):
    policy_model = PolicyModel2D(robot_trans_model, objsearch_reward_model)

    starget = ObjectState2D(target_id, "target", (4, 5))
    state = CosState2D({robot_id: init_srobot,
                        target_id: starget})
    assert policy_model.valid_moves(state) == ALL_MOVES_2D
    assert policy_model.get_all_actions(state) == ALL_MOVES_2D | {Done()}

    robot_pose = (14, 5, 0)
    srobot = RobotState2D(robot_id, robot_pose)
    state = CosState2D({robot_id: srobot,
                        target_id: starget})
    assert policy_model.valid_moves(state) == ALL_MOVES_2D - {MoveAhead}

    robot_pose = (14, 5, 90)
    srobot = RobotState2D(robot_id, robot_pose)
    state = CosState2D({robot_id: srobot,
                        target_id: starget})
    assert policy_model.valid_moves(state) == ALL_MOVES_2D

    robot_pose = (14, 14, 90)
    srobot = RobotState2D(robot_id, robot_pose)
    state = CosState2D({robot_id: srobot,
                        target_id: starget})
    assert policy_model.valid_moves(state) == ALL_MOVES_2D - {MoveAhead}

    robot_pose = (14, 15, 90)  # Illegal robot pose, out of bound.
    srobot = RobotState2D(robot_id, robot_pose)
    state = CosState2D({robot_id: srobot,
                        target_id: starget})
    assert policy_model.valid_moves(state) == set()
