import math
import random
from pomdp_py import RolloutPolicy, ActionPrior
from .action import Move, Search, Done, MOVES_2D_GRID
from ..utils.math import euclidean_dist
from ..thor import constants

class PolicyModel2D(RolloutPolicy):
    def __init__(self, robot_trans_model, reward_model,
                 num_visits_init=10, val_init=constants.TOS_REWARD_HI):
        self.robot_trans_model = robot_trans_model
        self._legal_moves = {}
        self._reward_model = reward_model
        self.action_prior = PolicyModel2D.ActionPrior(num_visits_init,
                                                     val_init, self)

    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state, history=None):
        return self.valid_moves(state) | {Done()}# + [Search(), Done()]

    def rollout(self, state, history=None):
        if self.action_prior is not None:
            preferences = self.action_prior.get_preferred_actions(state, history)\
                | {(Done(), 0, 0)}
            if len(preferences) > 0:
                return random.sample(preferences, 1)[0][0]
            else:
                return random.sample(self.get_all_actions(state=state), 1)[0]
        else:
            return random.sample(self.get_all_actions(state=state), 1)[0]

    def valid_moves(self, state):
        if state.robot_state in self._legal_moves:
            return self._legal_moves[state.robot_state]
        else:
            robot_pose = state.robot_state["pose"]
            valid_moves = set(a for a in MOVES_2D_GRID
                if self.robot_trans_model.sample(state, a)["pose"] != robot_pose)
            self._legal_moves[state.robot_state] = valid_moves
            return valid_moves

    class ActionPrior(ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            robot_state = state.robot_state
            preferences = set()
            target_loc = state.target_state["loc"]
            current_dist = euclidean_dist(
                state.robot_state["pose"][:2], target_loc)
            target_angle = (math.atan2(target_loc[1] - robot_state["pose"][1],
                                       target_loc[0] - robot_state["pose"][0])) % (360.0)
            cur_angle_diff = abs(robot_state["pose"][2] - target_angle)
            for move in MOVES_2D_GRID:
                next_robot_state = self.policy_model.robot_trans_model.sample(state, move)
                if euclidean_dist(next_robot_state["pose"][:2], target_loc) < current_dist:
                    preferences.add((move, self.num_visits_init, self.val_init))
                else:
                    next_angle_diff = abs(next_robot_state["pose"][2] - target_angle)
                    if next_angle_diff < cur_angle_diff:
                else:
                    preferences.add((move, self.num_visits_init, self.val_init / 2))
            return preferences
