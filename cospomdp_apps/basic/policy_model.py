# This instantiates the policy model used in the COS-POMDP
# created for this domain. Note that this policy model
# is used for planning at the COS-POMDP level.
import random
import cospomdp
from .action import Move2D, ALL_MOVES_2D, Done
from pomdp_py import RolloutPolicy, ActionPrior
from cospomdp.utils.math import euclidean_dist
from cospomdp.models.sensors import yaw_facing

############################
# Policy Model
############################
class PolicyModel2D(cospomdp.PolicyModel):
    def __init__(self, robot_trans_model,
                 reward_model,
                 movements=ALL_MOVES_2D,
                 **kwargs):
        super().__init__(robot_trans_model,
                         **kwargs)
        self._legal_moves = {}
        self.movements = movements
        self.reward_model = reward_model

    def set_observation_model(self, observation_model,
                              use_heuristic=True):
        super().set_observation_model(observation_model)
        if use_heuristic:
            self.action_prior = PolicyModel2D.ActionPrior(self.num_visits_init,
                                                          self.val_init, self)

    def get_all_actions(self, state, history=None):
        return self.valid_moves(state) | {Done()}# + [Search(), Done()]

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot["pose"]
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
            # If you have taken done before, you are done. So keep the done.
            last_action = history[-1][0] if len(history) > 0 else None
            if isinstance(last_action, Done):
                return {(Done(), 0, 0)}

            preferences = set()

            robot_id = self.policy_model.robot_id
            target_id = self.policy_model.observation_model.target_id
            srobot = state.s(robot_id)
            starget = state.s(target_id)
            if self.policy_model.reward_model.success(srobot, starget):
                preferences.add((Done(), self.num_visits_init, self.val_init))

            current_distance = euclidean_dist(srobot.loc, starget.loc)
            desired_yaw = yaw_facing(srobot.loc, starget.loc)
            current_yaw_diff = abs(desired_yaw - srobot.pose[2]) % 360

            for move in self.policy_model.movements:
                # A move is preferred if:
                # (1) it moves the robot closer to the target
                next_srobot = self.policy_model.robot_trans_model.sample(state, move)
                next_distance = euclidean_dist(next_srobot.loc, starget.loc)
                if next_distance < current_distance:
                    preferences.add((move, self.num_visits_init, self.val_init))
                    break

                # (2) if the move rotates the robot to be more facing the target,
                # unless the previous move was a rotation in an opposite direction;
                next_yaw_diff = abs(desired_yaw - next_srobot.pose[2]) % 360
                if next_yaw_diff < current_yaw_diff:
                    if hasattr(last_action, "dyaw") and last_action.dyaw * move.dyaw >= 0:
                        # last action and current are NOT rotations in different directions
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break

                # (3) it makes the robot observe any object;
                next_state = cospomdp.CosState({target_id: state.s(target_id),
                                                robot_id: next_srobot})
                observation = self.policy_model.observation_model.sample(next_state, move)
                for zi in observation:
                    if zi.loc is not None:
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break
            return preferences
