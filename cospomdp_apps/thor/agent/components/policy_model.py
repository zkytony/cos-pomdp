import random
from pomdp_py import RolloutPolicy, ActionPrior
from cospomdp.domain.action import Done
from cospomdp.utils.math import euclidean_dist
import cospomdp
from .action import MoveTopo, Stay

class PolicyModelTopo(cospomdp.PolicyModel):

    def __init__(self,
                 robot_trans_model,
                 topo_map, **kwargs):
        super().__init__(robot_trans_model, **kwargs)
        self._legal_moves = {}
        self._topo_map = topo_map

    def set_observation_model(self, observation_model,
                              use_heuristic=True):
        super().set_observation_model(observation_model)
        if use_heuristic:
            self.action_prior = PolicyModelTopo.ActionPrior(self.num_visits_init,
                                                            self.val_init,
                                                            self)

    @property
    def topo_map(self):
        return self._topo_map

    @property
    def robot_id(self):
        return self.robot_trans_model.robot_id

    @property
    def target_id(self):
        return self._observation_model.target_id

    def get_all_actions(self, state, history=None):
        return self.valid_moves(state) | {Done()}

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot["pose"]
            valid_moves = {Stay(srobot.nid)}  # stay is always a valid 'move'
            for nb_id in self._topo_map.neighbors(srobot.nid):
                eid = self._topo_map.edge_between(srobot.nid, nb_id)
                valid_moves.add(MoveTopo(srobot.nid,
                                         nb_id,
                                         self._topo_map.edges[eid].grid_dist))
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    def update(self, topo_map):
        """Update the topo_map"""
        self._topo_map = topo_map
        self._legal_moves = {}  # clear

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

            for move in self.policy_model.valid_moves(state):
                next_srobot = self.policy_model.robot_trans_model.sample(state, move)
                next_state = cospomdp.CosState({starget.id: state.s(starget.id),
                                                srobot.id: next_srobot})
                observation = self.policy_model.observation_model.sample(next_state, move)
                for zi in observation:
                    if zi.loc is not None:
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break

            if len(preferences) == 0:
                preferences.add((Stay(srobot.nid), self.num_visits_init, self.val_init))

            # # OLD
            # closest_target_nid = topo_map.closest_node(*starget.loc)
            # path = topo_map.shortest_path(srobot.nid, closest_target_nid)
            # current_gdist = sum(topo_map.edges[eid].grid_dist for eid in path)
            # for move in self.policy_model.valid_moves(state):
            #     path = topo_map.shortest_path(move.dst_nid, closest_target_nid)
            #     next_gdist = sum(topo_map.edges[eid].grid_dist for eid in path)
            #     if next_gdist < current_gdist:
            #         preferences.add((move, self.num_visits_init, self.val_init))
            return preferences | {(Done(), 0, 0)}
