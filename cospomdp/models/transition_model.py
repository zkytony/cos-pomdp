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

import math
from pomdp_py import TransitionModel

from ..utils.math import indicator, to_rad, fround
from ..domain.state import RobotState, CosState, ObjectState, RobotStatus

class RobotTransition(TransitionModel):
    """Models Pr(sr' | s, a); Likely domain-specific"""
    def __init__(self, robot_id):
        self.robot_id = robot_id


class CosTransitionModel(TransitionModel):
    """Cos-POMDP transition model Pr(s' | s, a)
    where the state s and s' are factored into robot and target states."""
    def __init__(self, target_id, robot_trans_model):
        self.target_id = target_id
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        starget = state.s(self.target_id)
        next_target_state = starget.copy()
        robot_id = self.robot_trans_model.robot_id
        return CosState({robot_id: next_robot_state,
                         self.target_id: next_target_state})


class FullTransitionModel(TransitionModel):
    """F-POMDP transition model Pr(s' | s, a)
    where the state s and s' are factored into robot and n object states."""
    def __init__(self, robot_trans_model):
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        objstates = {next_robot_state.id:next_robot_state}
        for objid in state.object_states:
            if objid == next_robot_state.id:
                continue
            next_object_state = state.s(objid).copy()
            objstates[objid] = next_object_state
        return CosState(objstates)
