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
import random
from thortils.navigation import find_navigation_plan, get_navigation_actions

import cospomdp
from cospomdp_apps import basic
from cospomdp_apps.thor.common import TOS_Action, Height
from cospomdp_apps.thor import constants
from cospomdp_apps.basic.belief import initialize_target_belief_2d, update_target_belief_2d
from cospomdp_apps.basic.action import Move2D
from cospomdp.utils.math import approx_equal, normalize, euclidean_dist
from .belief import initialize_target_belief_3d, update_target_belief_3d
from .action import (MoveTopo, Stay, Move,
                     from_grid_action_to_thor_action_params,
                     from_thor_delta_to_thor_action_params,
                     grid_navigation_actions2d,
                     grid_navigation_actions,
                     grid_camera_look_actions,
                     grid_height_range,
                     grid_pitch)
from .state import ObjectState3D, RobotState3D
from .transition_model import RobotTransition3D
from .policy_model import PolicyModel3D
from ..cospomdp_basic import ThorObjectSearchBasicCosAgent



# -------- Goal Handlers -------- #
class GoalHandler:
    def __init__(self, goal, agent):
        """
        agent: ThorObjectSearchCompleteCosAgent
        """
        self.goal = goal

    def step(self):
        raise NotImplementedError

    def update(self, tos_action, tos_observation):
        raise NotImplementedError

    @property
    def done(self):
        raise NotImplementedError

    def __str__(self):
        return "({}, done={})".format(self.goal, self.done)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self)

    @property
    def ispomdp(self):
        raise NotImplementedError

    @property
    def updates_first(self):
        """True if the handler should be updated before
        the COSPOMDP agent is updated"""
        return True


class DummyGoalHandler(GoalHandler):
    """A skeleton. Used for replay"""
    def __init__(self, goal, goal_done, agent):
        """
        agent: ThorObjectSearchCompleteCosAgent
        """
        self.goal = goal
        self._goal_done = goal_done

    def step(self):
        pass

    def update(self, tos_action, tos_observation):
        pass

    @property
    def done(self):
        return True


class MacroMoveHandler(GoalHandler):
    def __init__(self, dest_pos, agent, rot=None, angle_tolerance=15, goal=None):
        """
        dest_pos (position of the destination); Assume to be a 2D grid position.
        rot (rotation): Assume to be in grid map coordinates
        """
        if goal is None:
            super().__init__(dest_pos, agent)
        else:
            super().__init__(goal, agent)

        # Plans a sequence of actions to go from where
        # the robot is currently to the dst node.
        robot_pose = agent.belief.random().s(agent.robot_id).pose

        # Preparing the inputs for find_navigation_plan in thortils
        thor_rx, thor_rz, thor_rth = agent.grid_map.to_thor_pose(*robot_pose)
        thor_start_position = (thor_rx, 0, thor_rz)
        thor_start_rotation = (0, thor_rth, 0)
        _goal_pos = dest_pos
        thor_gx, thor_gz = agent.grid_map.to_thor_pos(*_goal_pos)
        thor_goal_position = (thor_gx, 0, thor_gz)
        if rot is None:
            thor_goal_rotation = (0, 0, 0)  # we don't care about rotation here
            angle_tolerance = 360
        else:
            thor_goal_rotation = (rot[0], agent.grid_map.to_thor_yaw(rot[1]), rot[2])
        thor_reachable_positions = [agent.grid_map.to_thor_pos(*p)
                                    for p in agent.reachable_positions]
        navigation_actions = get_navigation_actions(agent.thor_movement_params)
        plan, _ = find_navigation_plan((thor_start_position, thor_start_rotation),
                                       (thor_goal_position, thor_goal_rotation),
                                       navigation_actions,
                                       thor_reachable_positions,
                                       grid_size=agent.grid_map.grid_size,
                                       diagonal_ok=agent.task_config["nav_config"]["diagonal_ok"],
                                       angle_tolerance=angle_tolerance,
                                       debug=True)
        self._plan = plan
        self._index = 0

    def step(self):
        try:
            action_name, action_delta = self._plan[self._index]["action"]
            params = from_thor_delta_to_thor_action_params(action_name, action_delta)
            return TOS_Action(action_name, params)
        except:
            return None

    def update(self, tos_action, tos_observation):
        if self._index >= len(self._plan):
            return  # the index is already out of bound. we are done

        # Check if the robot pose is expected
        thor_rx = tos_observation.robot_pose[0]['x']
        thor_rz = tos_observation.robot_pose[0]['z']
        thor_rth = tos_observation.robot_pose[1]['y']
        actual = (thor_rx, thor_rz, thor_rth)
        expected_thor_rx, expected_thor_rz, _, expected_thor_rth = self._plan[self._index]['next_pose']
        expected = (expected_thor_rx, expected_thor_rz, expected_thor_rth)
        if not approx_equal(expected, actual, epsilon=1e-4):
            print("Warning: after taking {}, the robot pose is unexpected.\n"
                  "The expected pose is: {}; The actual pose is: {}"\
                  .format(tos_action, expected, actual))
        self._index += 1

    @property
    def done(self):
        if self._plan:
            return self._index >= len(self._plan)
        else:
            return True

    @property
    def ispomdp(self):
        return False


class MoveTopoHandler(MacroMoveHandler):
    """Deals with navigating along an edge on the topological map."""
    #
    def __init__(self, goal, agent):
        assert isinstance(goal, MoveTopo)
        super().__init__(agent.topo_map.nodes[goal.dst_nid].pos,
                         agent, rot=None, goal=goal) # we don't care about rotation here


class DoneHandler(GoalHandler):
    def __init__(self, goal, agent):
        super().__init__(goal, agent)

    def step(self):
        return TOS_Action(cospomdp.Done().name, {})

    def update(self, tos_action, tos_observation):
        pass

    @property
    def done(self):
        return True

    @property
    def ispomdp(self):
        return False


class LocalSearchHandler(GoalHandler):
    """Invoked when the COS-POMDP agent decides to 'Stay'.  There is potentially a
    lot of extensions to this handler, depending on whether you would like to
    consider 3D, object interaction, occlusion etc.
    """
    @staticmethod
    def create(goal, agent, search_type, params):
        """
        Factory function.
        search_type determines which kind of local searcher to use.
        This can be:

            basic (uses the basic COS-POMDP locally i.e. with low-level actions)
            3D (uses COS-POMDP locally with low-level actions, yet with a
                3D state, action and observation spaces) TODO: not yet implemented
            3D_interact (represents the search problem in 3D while
                considering local interactions with objects, particularly containers)

        Args:
            goal: should be a Stay
            agent (ThorObjectSearchCosAgent): the COS-POMDP agent that plans goals.
        """
        assert isinstance(goal, Stay)
        if search_type == "basic":
            return LocalSearchBasicHandler(goal, agent, params)
        elif search_type == "3d":
            return LocalSearch3DHandler(goal, agent, params)
        else:
            raise NotImplemented("Search type {} is not yet implemented".format(search_type))

    @property
    def ispomdp(self):
        return True

    @property
    def updates_first(self):
        """True if the handler should be updated before
        the COSPOMDP agent is updated"""
        return False

    @property
    def cos_agent(self):
        return self._local_cos_agent

    @property
    def detectable_objects(self):
        return self.cos_agent.detectable_objects

    @property
    def grid_map(self):
        return self._parent.grid_map

    @property
    def done(self):
        return self._done


class LocalSearchBasicHandler(LocalSearchHandler, ThorObjectSearchBasicCosAgent):
    """Even though the search is expected to be local,
    we don't manually restrict a region for the agent
    to operate - the size of the search region is the same,
    it's just the actions are smaller."""
    def __init__(self, goal, agent, params):
        """
        agent (ThorObjectSearchCosAgent)
        """
        LocalSearchHandler.__init__(self, goal, agent)
        self._parent = agent
        local_robot_state = agent.robot_state()

        self.robot_id = agent.robot_id
        self.target_id = agent.robot_id

        self.search_region = agent.search_region
        reachable_positions = agent.reachable_positions

        movement_params = agent.task_config["nav_config"]["movement_params"]
        self.navigation_actions = grid_navigation_actions2d(movement_params,
                                                       agent.grid_map.grid_size)
        robot_trans_model = basic.RobotTransition2D(self.robot_id, reachable_positions)
        reward_model = agent.cos_agent.reward_model # the reward model is the same
        policy_model = basic.PolicyModel2D(robot_trans_model, reward_model,
                                           movements=self.navigation_actions)

        _btarget = agent.belief.b(self.target_id)
        prior = {s.loc : _btarget[s] for s in _btarget}
        self._local_cos_agent = cospomdp.CosAgent(agent.target,
                                                  local_robot_state,
                                                  self.search_region,
                                                  robot_trans_model,
                                                  policy_model,
                                                  agent.corr_dists,
                                                  agent.detectors,
                                                  reward_model,
                                                  initialize_target_belief_2d,
                                                  update_target_belief_2d,
                                                  prior=prior)
        pouct_params = params.get("pouct", params)  # backwards compatibliity
        self.solver = pomdp_py.POUCT(**pouct_params,
                                     rollout_policy=self._local_cos_agent.policy_model)
        self._done = False


    def step(self):
        print("Planning locally")
        action = self.solver.plan(self._local_cos_agent)
        #### DEBUGGING TREE #####
        dd = pomdp_py.TreeDebugger(self._local_cos_agent.tree)
        print(dd)
        #########################
        print("     Num Sims:", self.solver.last_num_sims)
        if isinstance(action, basic.Move2D):
            params = from_grid_action_to_thor_action_params(
                action, self._parent.grid_map.grid_size)
        else:
            params = {}
        return TOS_Action(action.name, params)

    def update(self, tos_action, tos_observation):
        """This local agent updates AFTER the COSPOMDP agent.
        Because the two both maintain belief at ground level,
        we can simply sync the beliefs"""
        # interpret low level action
        action = self.interpret_action(tos_action)
        observation = self.interpret_observation(tos_observation)
        self.solver.update(self._local_cos_agent, action, observation)

        self._local_cos_agent.set_belief(self._parent.belief)
        self._done = tos_action.name.lower() == "done"


class LocalSearch3DHandler(LocalSearchBasicHandler, ThorObjectSearchBasicCosAgent):
    """
    Every time the local searcher is created, it inherits the
    belief from the COS-POMDP about the object's 2D location. Then, it initializes
    its own belief about the target's height. This belief is only useful locally.

    In fact, you do not need to maintain belief about continuous height;
    You just need to estimate whether the object is ABOVE, BELOW or AT_THE_SAME_LEVEL
    with respect to the robot.
    """
    def __init__(self, goal, agent, params):
        """
        Args:
            agent (ThorObjectSearchCompleteCosAgent)
        """
        LocalSearchHandler.__init__(self, goal, agent)
        self._parent = agent
        parent_robot_state = agent.robot_state()
        local_robot_state = RobotState3D(parent_robot_state.id,
                                         parent_robot_state.pose,
                                         parent_robot_state.height,
                                         parent_robot_state.horizon,
                                         parent_robot_state.status)

        self.robot_id = agent.robot_id
        self.target_id = agent.target_id

        self.search_region = agent.search_region
        reachable_positions = agent.reachable_positions

        movement_params = agent.task_config["nav_config"]["movement_params"]
        self.navigation_actions = grid_navigation_actions(movement_params,
                                                          agent.grid_map.grid_size)
        self.camera_look_actions = grid_camera_look_actions(movement_params)
        v_angles = [grid_pitch(va) for va in agent.task_config['nav_config']['v_angles']]
        robot_trans_model = RobotTransition3D(self.robot_id, reachable_positions, v_angles)
        reward_model = agent.cos_agent.reward_model
        policy_model = PolicyModel3D(robot_trans_model, reward_model,
                                     movements=self.navigation_actions,
                                     camera_looks=self.camera_look_actions)

        _btarget = agent.belief.b(self.target_id)
        prior_loc = {s.loc: _btarget.loc_belief[s] for s in _btarget}
        prior_height = _btarget.height_belief
        self._local_cos_agent = cospomdp.CosAgent(agent.target,
                                                  local_robot_state,
                                                  self.search_region,
                                                  robot_trans_model,
                                                  policy_model,
                                                  agent.corr_dists,
                                                  agent.detectors,
                                                  reward_model,
                                                  initialize_target_belief_3d,
                                                  update_target_belief_3d,
                                                  prior=(prior_loc, prior_height),
                                                  binit_args={"grid_size": agent.grid_map.grid_size})
        pouct_params = params.get("pouct", params)  # backwards compatibility
        self.solver = pomdp_py.POUCT(**pouct_params,
                                     rollout_policy=self._local_cos_agent.policy_model)
        self._done = False

    def step(self):
        print("Planning locally")
        action = self.solver.plan(self._local_cos_agent)
        #### DEBUGGING TREE #####
        dd = pomdp_py.TreeDebugger(self._local_cos_agent.tree)
        print(dd)
        #########################
        print("     Num Sims:", self.solver.last_num_sims)
        if isinstance(action, Move) or isinstance(action, Move2D):
            params = from_grid_action_to_thor_action_params(
                action, self._parent.grid_map.grid_size)
        else:
            params = {}
        return TOS_Action(action.name, params)

    def update(self, tos_action, tos_observation):
        """This local agent updates AFTER the COSPOMDP agent.
        Because the two both maintain belief at ground level,
        we can simply sync the beliefs"""
        # interpret low level action
        action = self.interpret_action(tos_action)
        observation = self.interpret_observation(tos_observation)
        self.solver.update(self._local_cos_agent, action, observation)

        self._local_cos_agent.set_belief(self._parent.belief)
        self._done = tos_action.name.lower() == "done"

    @property
    def primitive_motions(self):
        return set(self.navigation_actions) | set(self.camera_look_actions)
