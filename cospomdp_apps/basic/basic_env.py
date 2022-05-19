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

import pomdp_py
from .transition_model import RobotTransition2D
from cospomdp.domain.state import (CosState, ObjectState)
from cospomdp.models.transition_model import FullTransitionModel

class BasicEnv2D(pomdp_py.Environment):
    """This is meant for providing a basic simulation environment
    for COS-POMDP. If you are using e.g. Thor, you may not need this."""
    def __init__(self, init_robot_state, objlocs, target_id,
                 reachable_positions, reward_model):
        objstates = {objid: ObjectState(objid, objid, objlocs[objid])
                     for objid in objlocs}
        init_state = CosState({**{init_robot_state.id:init_robot_state},
                                 **objstates})
        robot_trans_model = RobotTransition2D(init_robot_state.id,
                                            reachable_positions)
        transition_model = FullTransitionModel(robot_trans_model)
        super().__init__(init_state, transition_model=transition_model,
                         reward_model=reward_model)
