# This instantiates the policy model used in the COS-POMDP
# created for this domain. Note that this policy model
# is used for planning at the COS-POMDP level.
import random
import cospomdp
from pomdp_py import RolloutPolicy, ActionPrior
from .action import ALL_MOVES_2D, Done

############################
# Policy Model
############################
class PolicyModel2D(cospomdp.PolicyModel):
    def __init__(self, robot_trans_model,
                 movements=ALL_MOVES_2D,
                 **kwargs):
        super().__init__(robot_trans_model,
                         **kwargs)
        self._legal_moves = {}
        self.movements = movements

    def set_observation_model(self, observation_model,
                              use_heuristic=True):
        super().set_observation_model(observation_model)
        if use_heuristic:
            self.action_prior = PolicyModel2D.ActionPrior(self.num_visits_init,
                                                          self.val_init, self)

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
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot["pose"]
            # TODO: technically, the action set should be passed in instead of hard-coded
            valid_moves = set(a for a in self.movements
                if self.robot_trans_model.sample(state, a)["pose"] != robot_pose)
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    class ActionPrior(ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            robot_id = self.policy_model.robot_id
            target_id = self.policy_model.observation_model.target_id
            srobot = state.s(robot_id)
            preferences = {(Done(), 0, 0)}
            for move in self.policy_model.movements:
                next_srobot = self.policy_model.robot_trans_model.sample(state, move)
                next_state = cospomdp.CosState({target_id: state.s(target_id),
                                                robot_id: next_srobot})
                observation = self.policy_model.observation_model.sample(next_state, move)
                for zi in observation:
                    if zi.loc is not None:
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break
            return preferences
