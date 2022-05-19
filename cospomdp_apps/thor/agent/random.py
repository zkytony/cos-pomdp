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

import random

from cospomdp.domain.action import Done
from cospomdp_apps.basic.action import Move2D

from ..common import TOS_Action, ThorAgent
from .components.action import (grid_navigation_actions2d,
                                from_grid_action_to_thor_action_params,
                                grid_camera_look_actions,
                                Move)

class ThorObjectSearchRandomAgent(ThorAgent):
    """
    Selects an action uniformly at random
    """

    AGENT_USES_CONTROLLER = False

    @property
    def detectable_objects(self):
        return [self.task_config["target"]]

    def __init__(self, task_config, grid_map, seed=1000):
        super().__init__(task_config)
        self.task_config = task_config
        self.rnd = random.Random(seed)
        self.grid_map = grid_map
        self.robot_id = task_config['robot_id']

        movement_params = self.task_config["nav_config"]["movement_params"]
        self.navigation_actions = set(grid_navigation_actions2d(movement_params,
                                                                grid_map.grid_size))
        self.camera_look_actions = set(grid_camera_look_actions(movement_params))

    def act(self):
        action = self.rnd.sample(self.navigation_actions | self.camera_look_actions | {Done()}, 1)[0]

        if not isinstance(action, TOS_Action):
            if isinstance(action, Move) or isinstance(action, Move2D):
                name = action.name
                params = from_grid_action_to_thor_action_params(action, self.grid_map.grid_size)
            elif isinstance(action, Done):
                name = "done"
                params = {}
            return TOS_Action(name, params)
        else:
            return action

    def update(self, tos_action, tos_observation):
        # Nothing to do.
        pass
