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
import math
import pomdp_py
import thortils as tt

from ..common import TOS_Action, ThorAgent
from .. import constants
from cospomdp.utils.math import indicator, normalize, euclidean_dist
from cospomdp_apps.thor.replay import ReplaySolver


# Used by optimal agent
import time
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)

from .components.action import from_thor_delta_to_thor_action_params


class ThorObjectSearchOptimalAgent(ThorAgent):
    """The agent find the shortest path to the target, and then
    follows this path by taking appropriate actions.
    """
    # Because this agent is not realistic, we permit it to have
    # access to the controller.
    AGENT_USES_CONTROLLER = True

    def __init__(self,
                 controller,
                 task_config):
        """Builds the agent and computes a plan"""
        self.controller = controller
        self.target = task_config["target"]
        self.task_type = task_config["task_type"]
        self.task_config = task_config

        start_pose = tt.thor_camera_pose(self.controller, as_tuple=True)
        overall_plan, overall_poses = ThorObjectSearchOptimalAgent.plan(
            self.controller, start_pose, self.target,
            self.task_type, **task_config["nav_config"])

        self.plan = overall_plan
        self._poses = overall_poses
        self._index = 0

    @property
    def detector(self):
        return None

    def act(self):
        """Returns action in plan"""
        if self._index < len(self.plan):
            action_name, action_params = self.plan[self._index]
            if action_name not in self.task_config["nav_config"]["movement_params"]:
                action = TOS_Action(action_name, action_params)
            else:
                delta = action_params
                thor_action_params = from_thor_delta_to_thor_action_params(action_name, delta)
                action = TOS_Action(action_name, thor_action_params)
            self._index += 1
        else:
            action = TOS_Action("Done", {})
        return action

    def update(self, action, observation):
        pass

    @classmethod
    def plan(cls, controller, start_pose, target, task_type, **nav_config):
        """
        Plans navigation + container opening actions
        where the navigation actions are movements along the shortest path
        found by A* on the grid map of environment.

        Args:
            controller (ai2thor.Controller)
            start_pose (tuple): position (x,y,z), rotation (pitch, yaw, roll).
                Both are tuples.
            target (str): Object type or ID
            task_type (str): "class" or "object"
            nav_config (dict): Navigation configurations
                (e.g. goal_distance, v_angles, h_angles, etc.), used in path finding.
                Refer to get_shortest_path_to_object for complete configuration.
        Returns:
            plan, poses

            plan: List of (action_name, params) tuples representing actions.
                The params are specific to the action; For movements, it's
                (forward, pitch, yaw). For opening, it is an object id.
            poses: List of (dict(x=,y=,z=), dict(x=,y=,z=)) (position, rotation) tuples.
                where each pose at index i corresponds to the pose resulting from taking
                action i.
        """
        start_position, start_rotation = start_pose
        if task_type == "class":
            object_type = target
            target_object = tt.thor_closest_object_of_type(controller, object_type)
        else:
            target_object_id = target
            target_object = tt.thor_object_with_id(controller, target_object_id)

        # Check if the target object is within openable receptacle. If so,
        # need to navigate to the receptacle, open it.
        # so receptors are also targets we want to navigate to
        ### NOTE: This method is hacky. It works for some objects, but has the following issues:
        ### 1. The robot doesn't know if it can successfully open the container
        ###    door (especially fridge).
        ### 2. An object, if placed very low, may be blocked by e.g. tabletop
        ###    and right now the robot cannot adjust its height to detect those
        ###    objects
        ### 3. (This is unrelated to this method, but it's an issue) For some
        ###    objects, their height seems wrong with respect to robot's height,
        ###    which causes the wrong pitch rotation to be computed
        ### THEREFORE, this optimal agent isn't complete; It returns a plan,
        ### which is approximately optimal (shortest), but it won't be able to
        ### execute it successfully 100% of the time because of the above issues.
        openable_receptors = tt.thor_object_receptors(controller,
                                                      target_object["objectId"],
                                                      openable_only=True)
        openable_receptor_ids = set(r["objectId"] for r in openable_receptors if r["openable"])

        goal_objects = openable_receptors + [target_object]
        overall_poses, overall_plan = [], []
        position, rotation = start_position, start_rotation
        for obj in goal_objects:
            _start_time = time.time()
            _nav_config = dict(nav_config)
            poses, plan = get_shortest_path_to_object(
                controller, obj["objectId"],
                position, rotation,
                return_plan=True,
                **nav_config
            )
            if plan is not None:
                print("plan found in {:.3f}s".format(time.time() - _start_time))
            else:
                raise ValueError("Plan not found to {}".format(obj["objectId"]))

            if len(poses) == 0:
                # No action needed. The robot is at where it should be already.
                assert len(plan) == 0
                pose_to_extend = tt.thor_pose_as_dict((position, rotation))
            else:
                pose_to_extend = poses[-1]
                # in case navigation is needed between containers (should be rare),
                # update the position and rotation for the next search.
                position, rotation = tt.thor_pose_as_tuple(poses[-1])

            # Check if we need an open action
            if obj["objectId"] in openable_receptor_ids:
                open_action = ("OpenObject", dict(objectId=obj["objectId"]))
                plan.append(open_action)
                poses.append(pose_to_extend)  # just so that both lists have same length

            overall_poses.extend(poses)
            overall_plan.extend(plan)


        if plan is None:
            raise ValueError("Plan to {} not found".format(target))

        return overall_plan, overall_poses
#############################################################################
