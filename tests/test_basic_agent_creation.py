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

import numpy as np
from cospomdp.domain.state import RobotState2D
from cospomdp.models.agent import CosAgent
from cospomdp.models.search_region import SearchRegion2D
from cospomdp.models.correlation import CorrelationDist
from cospomdp.models.observation_model import (FanModelYoonseon,
                                               FanModelNoFP)
from cospomdp.models.reward_model import ObjectSearchRewardModel, NavRewardModel
from cospomdp_apps.basic import RobotTransition2D, PolicyModel2D, initialize_target_belief_2d, update_target_belief_2d

def test_agent_creation():

    robot_id = 0
    init_robot_pose = (1,1,0)

    mug = (1, "Mug")
    table = (2, "Table")
    coffeemachine = (3, "CoffeeMachine")
    target = mug

    locations = np.unique(np.random.randint(0, 5, size=(1000, 2)), axis=0)
    locations = list(map(tuple, locations.tolist()))
    search_region = SearchRegion2D(locations)

    reachable_positions = locations
    corr_dists = {
        table[0]: CorrelationDist(table, target, search_region, {(3,4):1.0, (2,2):1.5, (0,2):0.5}),
        coffeemachine[0]: CorrelationDist(coffeemachine, target, search_region, {(2,4):0.8, (1,2):1.1, (2,2):0.7}),
    }
    fan_params = dict(fov=90, min_range=0, max_range=3)
    detectors = {
        mug[0]: FanModelNoFP(mug[0], fan_params, (0.7, 0.1), round_to=None),
        table[0]: FanModelNoFP(table[0], fan_params, (0.8, 0.1), round_to=None),
        coffeemachine[0]: FanModelNoFP(coffeemachine[0], fan_params, (0.9, 0.1), round_to=None)
    }
    reward_model = ObjectSearchRewardModel(detectors[target[0]].sensor,
                                           1.5, robot_id, target[0])
    init_robot_state = RobotState2D(robot_id, init_robot_pose)
    robot_trans_model = RobotTransition2D(robot_id, reachable_positions)
    policy_model = PolicyModel2D(robot_trans_model, reward_model)
    agent = CosAgent(target,
                     init_robot_state,
                     search_region,
                     robot_trans_model,
                     policy_model,
                     corr_dists,
                     detectors,
                     reward_model,
                     initialize_target_belief_2d,
                     update_target_belief_2d)
    assert agent.belief.mpe().s(robot_id)["pose"] == init_robot_pose
