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

# This is a toy domain for 2D COS-POMDP
import pytest
import pomdp_py
import numpy as np
from cospomdp_apps.basic.action import ALL_MOVES_2D, Done
from cospomdp_apps.basic.parser import create_instance

@pytest.fixture
def world():
    WORLD =\
"""
### map
R....
.x.Tx
.xG.x

### robotconfig
th: 0

### corr
T around G: d=2

### detectors
T: fan-nofp | fov=45, min_range=0, max_range=2 | (0.6, 0.1)
G: fan-nofp | fov=45, min_range=0, max_range=3 | (0.8, 0.1)

### goal
find: T, 2.0

### END
"""
    return WORLD

def test_agent_creation_and_plan_from_parse(world):
    agent, objlocs = create_instance(world)
    planner = pomdp_py.POUCT(max_depth=10, discount_factor=0.95,
                             planning_time=.2, exploration_const=100,
                             rollout_policy=agent.policy_model)
    action = planner.plan(agent)
    assert action in ALL_MOVES_2D | {Done()}
