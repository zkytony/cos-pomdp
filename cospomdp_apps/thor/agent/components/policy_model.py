import random
from pomdp_py import RolloutPolicy, ActionPrior
from cospomdp.domain.action import Done
from cospomdp.models.sensors import pitch_facing, yaw_facing
from cospomdp.utils.math import euclidean_dist
from cospomdp_apps.basic.policy_model import PolicyModel2D
import cospomdp
from .action import MoveTopo, Stay

class PolicyModelTopo(cospomdp.PolicyModel):

    def __init__(self,
                 robot_trans_model, reward_model,
                 topo_map, **kwargs):
        super().__init__(robot_trans_model, **kwargs)
        self._legal_moves = {}
        self._topo_map = topo_map
        self.reward_model = reward_model

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
        all_actions = self.valid_moves(state) | {Done()}
        return all_actions

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
            # If you have taken done before, you are done. So keep the done.
            last_action = history[-1][0] if len(history) > 0 else None
            if isinstance(last_action, Done):
                return {(Done(), 0, 0)}

            preferences = set()

            topo_map = self.policy_model.topo_map
            srobot = state.s(self.policy_model.robot_id)
            starget = state.s(self.policy_model.target_id)

            if self.policy_model.reward_model.success(srobot, starget):
                preferences.add((Done(), self.num_visits_init, self.val_init))

            closest_target_nid = topo_map.closest_node(*starget.loc)
            path = topo_map.shortest_path(srobot.nid, closest_target_nid)
            current_gdist = sum(topo_map.edges[eid].grid_dist for eid in path)
            for move in self.policy_model.valid_moves(state):
                # A move is preferred if:
                # (1) it moves the robot closer to the target, in terms of geodesic distance
                next_path = topo_map.shortest_path(move.dst_nid, closest_target_nid)
                next_gdist = sum(topo_map.edges[eid].grid_dist for eid in next_path)
                if next_gdist < current_gdist:
                    preferences.add((move, self.num_visits_init, self.val_init))
                    break

                # (2) it is a stay, while any object can be observed after the
                # stay transition (which sets the facing yaw too)
                if isinstance(move, Stay):
                    next_srobot = self.policy_model.robot_trans_model.sample(state, move)
                    next_state = cospomdp.CosState({starget.id: state.s(starget.id),
                                                    srobot.id: next_srobot})
                    observation = self.policy_model.observation_model.sample(next_state, move)
                    for zi in observation:
                        if zi.loc is not None:
                            preferences.add((move, self.num_visits_init, self.val_init))
                            break
            return preferences


class PolicyModel3D(cospomdp.PolicyModel):
    def __init__(self, robot_trans_model, reward_model, movements, camera_looks, **kwargs):
        super().__init__(robot_trans_model, **kwargs)
        self._legal_moves = {}
        self.movements = set(movements)
        self.camera_looks = set(camera_looks)
        self.reward_model = reward_model

    def set_observation_model(self, observation_model,
                              use_heuristic=True):
        super().set_observation_model(observation_model)
        if use_heuristic:
            self.action_prior = PolicyModel3D.ActionPrior(self.num_visits_init,
                                                          self.val_init, self)

    @property
    def primitive_motions(self):
        return self.movements | self.camera_looks

    def get_all_actions(self, state, history=None):
        all_actions = self.valid_moves(state) | {Done()}
        if len(all_actions) == 1:
            import pdb; pdb.set_trace()
        return all_actions

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot.pose3d
            valid_moves = set()
            LookUpin = False
            LookDownin = False
            for a in self.primitive_motions:
                if self.robot_trans_model.sample(state, a).pose3d != robot_pose:
                    valid_moves.add(a)
            # valid_moves = set(a for a in self.primitive_motions
            #     if self.robot_trans_model.sample(state, a).pose3d != robot_pose)
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    class ActionPrior(ActionPrior):
        """Reuse some of the ActionPrior in 2D"""
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model
            self._cache = {}

        def get_preferred_actions(self, state, history):
            if state in self._cache:
                return self._cache[state]

            last_action = history[-1][0] if len(history) > 0 else None
            if isinstance(last_action, Done):
                return {(Done(), 0, 0)}

            robot_id = self.policy_model.robot_id
            target_id = self.policy_model.observation_model.target_id
            srobot = state.s(robot_id)
            starget = state.s(target_id)

            preferences = set()

            if self.policy_model.reward_model.success(srobot, starget):
                preferences.add((Done(), self.num_visits_init, self.val_init))

            # This borows from Policy2D
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

            self._cache[state] = preferences
            return preferences
