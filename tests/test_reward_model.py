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

import pytest
from cospomdp.models import *
from cospomdp.domain import *
from cospomdp_apps.basic import RobotTransition2D
from cospomdp_apps.basic.action import *

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
    T = CosTransitionModel(target_id, Trobot)
    R = ObjectSearchRewardModel(fansensor, 2.0, robot_id, target_id)

    starget = ObjectState(target_id, "target", (4, 5))
    state = CosState({robot_id: init_srobot,
                        target_id: starget})
    assert R.sample(state, Done(), T.sample(state, Done())) == R._hi
    assert R.sample(state, MoveAhead, T.sample(state, MoveAhead)) == R._step

    state2 = T.sample(T.sample(state, RotateLeft), RotateLeft)
    assert R.sample(state2, Done(), T.sample(state2, Done())) == R._lo


def test_reward_model_nav(init_srobot, search_region):
    robot_id = 0
    target_id = 10

    Trobot = RobotTransition2D(robot_id, search_region.locations)
    T = CosTransitionModel(target_id, Trobot)
    R = NavRewardModel((2, 6, 135), robot_id)
    starget = ObjectState(target_id, "target", (4, 5))
    state = CosState({robot_id: init_srobot,
                      target_id: starget})

    R.sample(state, Done(), T.sample(state, Done())) == R._lo

    state2 = T.sample(T.sample(T.sample(T.sample(state, RotateLeft), RotateLeft), MoveAhead), RotateLeft)
    R.sample(state2, Done(), T.sample(state2, Done())) == R._lo
    R.sample(state2, MoveAhead, T.sample(state, MoveAhead)) == R._step
