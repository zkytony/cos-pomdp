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

"""
Overall control flow

     ThorObjectSearchCosAgent --> act --> ThorObjectSearch --\
    /               (solver.plan(COSAgent))                   \
   \ (CosAgent.update())                                       \
    \                                                           \
     \                       agent                                 execute
      <--------------------- update ------------- observation /

"""

import random
import time
import math
import numpy as np
import os
import pomdp_py
import thortils as tt

import cospomdp
from cospomdp.utils.math import indicator, normalize, euclidean_dist, roundany, closest
from cospomdp_apps.basic import PolicyModel2D, RobotTransition2D
from cospomdp_apps.basic.action import Move2D, ALL_MOVES_2D, Done
from cospomdp_apps.basic.belief import initialize_target_belief_2d, update_target_belief_2d

from .components.action import (grid_navigation_actions2d,
                                from_grid_action_to_thor_action_params)
from .components.state import grid_full_pose

from ..common import TOS_Action, ThorAgent
from ..replay import ReplaySolver
from .. import constants
from .. import paths


class GridMapSearchRegion(cospomdp.SearchRegion2D):
    def __init__(self, grid_map, scene=None):
        super().__init__(grid_map.obstacles)
        self.scene = scene


class ThorObjectSearchCosAgent(ThorAgent):
    AGENT_USES_CONTROLLER = False

    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 grid_map,
                 thor_camera_pose):
        """
        Initialize.
        """
        super().__init__(task_config)
        self.robot_id = task_config["robot_id"]
        scene = task_config["scene"]
        search_region = GridMapSearchRegion(grid_map, scene=scene)
        reachable_positions = grid_map.free_locations
        self.grid_map = grid_map
        self.search_region = search_region
        self.reachable_positions = reachable_positions

        x, y, height, pitch, yaw = grid_full_pose(thor_camera_pose,
                                                  task_config['nav_config']['v_angles'],
                                                  grid_map)

        self._init_robot_pose = (x, y, yaw)   # 2D pose
        self._init_pitch = pitch
        self._height = height

        if task_config["task_type"] == 'class':
            target_id = task_config['target']
            target_class = task_config['target']
            target = (target_id, target_class)
        else:
            # This situation is not tested :todo:
            target = task_config['target']  # (target_id, target_class)
            target_id = target[0]
        self.task_config = task_config
        self.target = target
        self.target_id = target[0]

        detectors, detectable_objects = self._build_detectors(detector_specs)

        # load correlation distributions
        corr_dists_path = None
        if task_config.get("save_load_corr", False):
            corr_dists_path = paths.CORR_DISTS_PATH
        corr_dists = self._build_corr_dists(corr_specs, detectable_objects,
                                            corr_dists_path=corr_dists_path)
        self.corr_dists = corr_dists
        self.detectors = detectors
        self.detectable_objects = detectable_objects

        self.thor_movement_params = task_config["nav_config"]["movement_params"]

    @property
    def belief(self):
        return self.cos_agent.belief

    def sensor(self, objid):
        return self.cos_agent.sensor(objid)

    def robot_state(self):
        """Since robot state is observable, return the mpe state in belief"""
        return self.cos_agent.belief.b(self.robot_id).mpe()

    def _build_detectors(self, detector_specs):
        return ThorObjectSearchCosAgent.build_detectors(
            self.task_config["detectables"], detector_specs)

    @staticmethod
    def build_detectors(detectables, detector_specs):
        detectable_objects = {}  # objects we care about are detectable objects
        detectors = {}
        for obj in detectables:
            if type(obj) == tuple and len(obj) == 2:
                object_id, object_class = obj
            else:
                object_id = object_class = obj
            detectable_objects[object_id] = (object_id, object_class)

            detector_type, sensor_params, quality_params = detector_specs[object_id]
            if detector_type.strip() in {"fan-nofp", "fan-simplefp", "fan-far"}:
                if type(sensor_params) == str:
                    sensor_params = eval(f"dict({sensor_params.strip()})")
                if quality_params == str:
                    quality_params = eval(quality_params.strip())
                if detector_type.strip() == "fan-nofp":
                    detector = cospomdp.FanModelNoFP(object_id, sensor_params, quality_params)
                elif detector_type.strip() == "fan-simplefp":
                    detector = cospomdp.FanModelSimpleFP(object_id, sensor_params, quality_params)
                else: # detector_type.strip() == "fan-far":
                    detector = cospomdp.FanModelFarRange(object_id, sensor_params, quality_params)
                detectors[object_id] = detector
        return detectors, detectable_objects

    def _build_corr_dists(self, corr_specs, objects, corr_dists_path):
        return ThorObjectSearchCosAgent.build_corr_dists(
            self.target[0], self.search_region,
            corr_specs, objects, corr_dists_path=corr_dists_path)

    @staticmethod
    def build_corr_dists(target_id, search_region, corr_specs, objects, corr_dists_path=None):
        """
        corr_dists_path (str): Path to the directory that contains corr_dists pickle files
        """
        target = objects[target_id]
        corr_dists = {}
        for key in corr_specs:
            obj1, obj2 = key
            if obj1 == target_id:
                other = obj2
            elif obj2 == target_id:
                other = obj1
            else:
                continue

            if other not in corr_dists:
                corr_object = objects[other]
                corr_func, corr_func_args = corr_specs[key]
                corr_type = corr_func_args.get("type", None)

                loaded = False
                if corr_dists_path is not None:
                    scene = search_region.scene
                    scene_type = tt.ithor_scene_type(scene)

                    cdist_path = os.path.join(
                        corr_dists_path, f"corr-dist_{scene_type}_{target[1]}-{corr_object[1]}_{scene}_{corr_type}.pkl")
                    if os.path.exists(cdist_path):
                        print(f"Loading corr dist of type '{corr_type}' Pr({corr_object[1]} | {target[1]})")
                        corr_dists[other] = cospomdp.CorrelationDist.load(cdist_path)
                        loaded = True

                if not loaded:
                    if type(corr_func) == str:
                        corr_func = eval(corr_func)
                    corr_dists[other] = cospomdp.CorrelationDist(
                        corr_object, target, search_region, corr_func,
                        corr_func_args=corr_func_args)

                    if corr_dists_path is not None:
                        # save
                        print(f"Saving corr dist of type '{corr_type}' Pr({corr_object[1]} | {target[1]}) to {cdist_path}")
                        os.makedirs(corr_dists_path, exist_ok=True)
                        corr_dists[other].save(cdist_path)

        return corr_dists

    def interpret_robot_obz(self, tos_observation):
        raise NotImplementedError

    def interpret_object_obzs(self, tos_observation):
        objobzs = {}
        for cls in self.detectable_objects:
            objobzs[cls] = cospomdp.Loc(cls, None)

        for detection in tos_observation.detections:
            xyxy, conf, cls, locs = detection

            # we will average locs to get an estimate of the
            # detected object's center, as this is the location
            # used in collecting distances for correlations.
            thor_x, _, thor_z = np.mean(np.asarray(locs), axis=0)

            x, z = self.grid_map.to_grid_pos(thor_x, thor_z)
            # we don't want to lose this detection because it is at 'unknown'.
            # so we will map it to the closest one
            if (x,z) not in self.search_region.locations:
                x, z = min(self.search_region.locations,
                           key=lambda l: euclidean_dist(l, (x,z)))
            objobzs[cls] = cospomdp.Loc(cls, (x, z))
        return objobzs

    def interpret_observation(self, tos_observation):
        objobzs = self.interpret_object_obzs(tos_observation)
        robotobz = self.interpret_robot_obz(tos_observation)
        observation = cospomdp.CosObservation(robotobz, objobzs)
        return observation

    def update(self, tos_action, tos_observation):
        """
        Given TOS_Action and TOS_Observation, update the agent's belief, etc.
        """
        # Because policy_model's movements are already in sync with task movements,
        # we can directly get the POMDP action from there.
        observation = self.interpret_observation(tos_observation)
        action = self.interpret_action(tos_action)
        self._update_belief(action, observation)

    def new_history(self, tos_action, tos_observation):
        return super().new_history(tos_action, tos_observation)


class ThorObjectSearchBasicCosAgent(ThorObjectSearchCosAgent):
    """
    The BasicCosAgent contains a COS-POMDP that is instantiated as in
    cospomdp_apps.basic; It uses a 2D grid map representation of the search
    region, and a 2D robot state representation. It directly outputs
    low-level movement actions such as MoveAhead; Therefore, there is
    no need for "goal interpretation" and the output is directly
    wrapped by a TOS_Action for execution.

    Note that this agent does not make use of LookUp / LookDown actions.
    That's why it is basic!
    """
    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 solver,
                 solver_args,
                 grid_map,
                 thor_camera_pose,
                 thor_prior={},
                 approx_belief=False):
        """
        controller (ai2thor Controller)
        task_config (dict) configuration; see make_config
        corr_specs (dict): Maps from (target_id, object_id) to (corr_func, corr_func_args)
        detector_specs (dict): Maps from object_id to (detector_type, sensor_params, quality_params)
        solver (str): name of solver
        solver_args (dict): arguments for the solver
        thor_prior: dict mapping from thor location to probability; If empty, then the prior will be uniform.
        """
        super().__init__(task_config,
                         corr_specs,
                         detector_specs,
                         grid_map,
                         thor_camera_pose)

        goal_distance = (task_config["nav_config"]["goal_distance"]
                         / self.grid_map.grid_size) * 0.8  # just to make sure we are close enough
        reward_model = cospomdp.ObjectSearchRewardModel(
            self.detectors[self.target_id].sensor, goal_distance,
            self.robot_id, self.target_id,
            **task_config["reward_config"])

        # Construct CosAgent, the actual POMDP
        init_robot_state = cospomdp.RobotState2D(self.robot_id, self._init_robot_pose)
        robot_trans_model = RobotTransition2D(self.robot_id, self.reachable_positions)
        movement_params = self.task_config["nav_config"]["movement_params"]
        self.navigation_actions = set(grid_navigation_actions2d(movement_params, grid_map.grid_size))
        policy_model = PolicyModel2D(robot_trans_model, reward_model,
                                     movements=self.navigation_actions)
        prior = {grid_map.to_grid_pos(p[0], p[2]): thor_prior[p]
                 for p in thor_prior}
        belief_type = "histogram" if not approx_belief else "histogram-approx"
        self.cos_agent = cospomdp.CosAgent(self.target, init_robot_state,
                                           self.search_region, robot_trans_model, policy_model,
                                           self.corr_dists, self.detectors, reward_model,
                                           initialize_target_belief_2d, update_target_belief_2d,
                                           prior=prior, belief_type=belief_type)
        # construct solver
        if solver == "pomdp_py.POUCT":
            self.solver = pomdp_py.POUCT(**solver_args,
                                         rollout_policy=self.cos_agent.policy_model)
        else:
            self.solver = eval(solver)(**solver_args)

    def act(self):
        """
        Output a TOS_Action
        """
        action = self.solver.plan(self.cos_agent)
        print("     Num Sims:", self.solver.last_num_sims)

        # Need to return TOS_Action
        if not isinstance(action, TOS_Action):
            if isinstance(action, Move2D):
                name = action.name
                params = self.thor_action_params(action)
            elif isinstance(action, Done):
                name = "done"
                params = {}
            return TOS_Action(name, params)
        else:
            return action

    def thor_action_params(self, action):
        """Different from external agents, which uses deep learning models that
        are trained with fixed rotation sizes, here we are able to set the parameters
        of ai2thor movements based on POMDP action's parameters. So we make sure
        they are in sync - i.e. we convert parameters in `action` to parameters
        that can be used for ai2thor.
        """
        return from_grid_action_to_thor_action_params(action, self.grid_map.grid_size)

    def interpret_robot_obz(self, tos_observation):
        thor_robot_pose = tos_observation.robot_pose
        thor_robot_pose2d = (thor_robot_pose[0]['x'], thor_robot_pose[0]['z'], thor_robot_pose[1]['y'])
        robot_pose = self.grid_map.to_grid_pose(*thor_robot_pose2d)
        return cospomdp.RobotObservation(self.robot_id,
                                         robot_pose,
                                         cospomdp.RobotStatus(done=tos_observation.done))

    def interpret_action(self, tos_action):
        cospomdp_actions = {a.name: a for a in set(self.primitive_motions) | {Done()}}
        if tos_action.name in cospomdp_actions:
            action = cospomdp_actions[tos_action.name]
            return action
        else:
            raise ValueError("Cannot understand action {}".format(action))

    def _update_belief(self, action, observation):
        """
        Here, action, observation are already interpreted.
        """
        self.cos_agent.update(action, observation)
        self.solver.update(self.cos_agent, action, observation)

    @property
    def primitive_motions(self):
        return self.navigation_actions
