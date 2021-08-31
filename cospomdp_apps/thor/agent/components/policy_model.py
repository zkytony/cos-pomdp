import random
from pomdp_py import RolloutPolicy, ActionPrior
from cospomdp.domain.action import Done
from cospomdp.utils.math import euclidean_dist
from .action import MoveTopo

class PolicyModelTopo(RolloutPolicy):

    def __init__(self,
                 robot_trans_model,
                 reward_model,
                 topo_map,
                 num_visits_init=10, val_init=100):

        self.robot_trans_model = robot_trans_model
        self._legal_moves = {}
        self._reward_model = reward_model
        self.topo_map = topo_map
        self.action_prior = PolicyModelTopo.ActionPrior(num_visits_init,
                                                        val_init,
                                                        self)

    @property
    def robot_id(self):
        return self.robot_trans_model.robot_id

    @property
    def target_id(self):
        return self._reward_model.target_id

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
            valid_moves = set()
            for nb_id in self.topo_map.neighbors(srobot.nid):
                eid = self.topo_map.edge_between(srobot.nid, nb_id)
                valid_moves.add(MoveTopo(srobot.nid,
                                         nb_id,
                                         self.topo_map.edges[eid].grid_dist))
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    def update(self):
        raise NotImplementedError


    class ActionPrior(ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            topo_map = self.policy_model.topo_map
            srobot = state.s(self.policy_model.robot_id)
            starget = state.s(self.policy_model.target_id)
            preferences = set()

            closest_target_nid = topo_map.closest_node(*starget.loc)
            path = topo_map.shortest_path(srobot.nid, closest_target_nid)
            current_gdist = sum(topo_map.edges[eid].grid_dist for eid in path)
            for move in self.policy_model.valid_moves(state):
                path = topo_map.shortest_path(move.dst_nid, closest_target_nid)
                next_gdist = sum(topo_map.edges[eid].grid_dist for eid in path)
                if next_gdist < current_gdist:
                    preferences.add((move, self.num_visits_init, self.val_init))

            if euclidean_dist(srobot.pose[:2],
                              starget.loc) <= self.policy_model._reward_model.goal_dist:
                preferences.add((Done(), self.num_visits_init, self.val_init))
            return preferences
