import math
import random
from pomdp_py import RolloutPolicy, ActionPrior
from ..domain.action import Done
from ..utils.math import euclidean_dist

class PolicyModel(RolloutPolicy):
    def __init__(self,
                 robot_trans_model,
                 all_actions,
                 num_visits_init=10,
                 val_init=100):
        self.robot_trans_model = robot_trans_model
        self.all_actions = all_actions
        self.action_prior = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self._observation_model = None  # this can be helpful for the action prior


    @property
    def robot_id(self):
        return self.robot_trans_model.robot_id


    @property
    def observation_model(self):
        return self._observation_model

    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state, history=None):
        raise NotImplementedError

    def rollout(self, state, history=None):
        if self.action_prior is not None:
            preferences = self.action_prior.get_preferred_actions(state, history)
            if len(preferences) > 0:
                return random.sample(preferences, 1)[0][0]
            else:
                return random.sample(self.get_all_actions(state=state), 1)[0]
        else:
            return random.sample(self.get_all_actions(state=state), 1)[0]

    def set_observation_model(self, observation_model):
        # Classes that inherit this class can override this
        # function to create action prior
        self._observation_model = observation_model
