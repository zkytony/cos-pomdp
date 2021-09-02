import pomdp_py
import time
import random
from collections import deque
import concurrent.futures

from thortils.navigation import find_navigation_plan, get_navigation_actions

from cospomdp.utils.math import euclidean_dist, normalize
import cospomdp

from ..constants import GOAL_DISTANCE
from ..common import ThorAgent, TOS_Action
from .cospomdp_basic import GridMapSearchRegion, ThorObjectSearchCosAgent
from .components.action import Move, MoveTopo, Stay
from .components.state import RobotStateTopo
from .components.topo_map import TopoNode, TopoMap, TopoEdge
from .components.transition_model import RobotTransitionTopo
from .components.policy_model import PolicyModelTopo
from .components.goal_handlers import (MoveTopoHandler,
                                       DoneHandler,
                                       LocalSearchHandler)


def _shortest_path(reachable_positions, gloc1, gloc2):
    """
    Computes the shortest distance between two locations.
    The two locations will be snapped to the closest free cell.
    """
    def neighbors(x,y):
        return [(x+1, y), (x-1,y),
                (x,y+1), (x,y-1)]

    def get_path(s, t, prev):
        v = t
        path = [t]
        while v != s:
            v = prev[v]
            path.append(v)
        return path

    # BFS; because no edge weight
    reachable_positions = set(reachable_positions)
    visited = set()
    q = deque()
    q.append(gloc1)
    prev = {gloc1:None}
    while len(q) > 0:
        loc = q.popleft()
        if loc == gloc2:
            return get_path(gloc1, gloc2, prev)
        for nb_loc in neighbors(*loc):
            if nb_loc in reachable_positions:
                if nb_loc not in visited:
                    q.append(nb_loc)
                    visited.add(nb_loc)
                    prev[nb_loc] = loc
    return None


def _sample_topo_map(target_hist,
                     reachable_positions,
                     num_samples,
                     degree=(3,5),
                     sep=4.0,
                     rnd=random,
                     robot_pos=None):
    """Given a search region, a distribution over target locations in the
    search region, return a TopoMap with nodes within
    reachable_positions.

    The algorithm works by first converting the target_hist,
    which is a distribution over the search region, to a distribution
    over the robot's reachable positions.

    This is done by, for each location in the search region, find
    a closest reachable position; Then the probability at a reachable
    position is the sum of those search region locations mapped to it.

    Then, simply sample reachable positions based on this distribution.

    The purpose of this topo map is for navigation action abstraction
    and robot state abstraction.

    Args:
        target_hist (dict): maps from location to probability
        reachable_positions (list of tuples)
        num_places (int): number of places to sample
        degree (int or tuple): Controls the minimum and maximum degree
            per topo node in the resulting graph. If only one number is
            passed, then will make all nodes have the same degree.
            This assumes there are enough sampled nodes to satisfy this
            requirement; If not, then all nodes are still guaranteed
            to have degree less than or equal to the maximum degree.
        sep (float): minimum distance between two places (grid cells)
        robot_pos (x,y): If not None, will add a node at where the robot is.

    Returns:
        TopologicalMap.

    """
    if type(degree) == int:
        degree_range = (degree, degree)
    else:
        degree_range = degree
        if len(degree_range) != 2:
            raise ValueError("Invalid argument for degree {}."
                             "Accepts int or (int, int)".format(degree))

    mapping = {}  # maps from reachable pos to a list of search region locs
    for loc in target_hist:
        closest_reachable_pos = min(reachable_positions,
                                    key=lambda robot_pos: euclidean_dist(loc, robot_pos))
        if closest_reachable_pos not in mapping:
            mapping[closest_reachable_pos] = []
        mapping[closest_reachable_pos].append(loc)

    # distribution over reachable positions
    reachable_pos_dist = {}
    for pos in mapping:
        reachable_pos_dist[pos] = 0.0
        for search_region_loc in mapping[pos]:
            reachable_pos_dist[pos] += target_hist[search_region_loc]
    hist = pomdp_py.Histogram(normalize(reachable_pos_dist))

    places = set()
    if robot_pos is not None:
        places.add(robot_pos)
    for i in range(num_samples):
        pos = hist.random(rnd=rnd)
        if len(places) > 0:
            closest_pos = min(places,
                              key=lambda c: euclidean_dist(pos, c))
            if euclidean_dist(closest_pos, pos) >= sep:
                places.add(pos)
        else:
            places.add(pos)

    # Create nodes
    pos_to_nid = {}
    nodes = {}
    for i, pos in enumerate(places):
        topo_node = TopoNode(i, pos, mapping.get(pos, set()))
        nodes[i] = topo_node
        pos_to_nid[pos] = i

    # Now, we need to connect the places to form a graph.
    _conns = {}
    edges = {}
    for nid in nodes:
        if nid not in _conns:
            _conns[nid] = set()
        neighbors = _conns[nid]
        neighbor_positions = {nodes[nbnid].pos for nbnid in neighbors}
        candidates = set(places) - {nodes[nid].pos} - neighbor_positions
        degree_needed = degree_range[0] - len(neighbors)
        if degree_needed <= 0:
            continue
        new_neighbors = list(sorted(candidates, key=lambda pos: euclidean_dist(pos, nodes[nid].pos)))[:degree_needed]
        for nbpos in new_neighbors:
            nbnid = pos_to_nid[nbpos]
            if nbnid not in _conns or len(_conns[nbnid]) < degree_range[1]:
                _conns[nid].add(nbnid)
                if nbnid not in _conns:
                    _conns[nbnid] = set()
                _conns[nbnid].add(nid)

                path = _shortest_path(reachable_positions,
                                      nodes[nbnid].pos,
                                      nodes[nid].pos)
                if path is None:
                    # Skip this edge because we cannot find path
                    continue
                eid = len(edges) + 1000
                edges[eid] = TopoEdge(eid,
                                      nodes[nid],
                                      nodes[nbnid],
                                      path)
    if len(edges) == 0:
        edges[0] = TopoEdge(0, nodes[next(iter(nodes))], None, [])

    topo_map = TopoMap(edges)
    # Verification
    for nid in topo_map.nodes:
        assert len(topo_map.edges_from(nid)) <= degree_range[1]

    return topo_map


class ThorObjectSearchCompleteCosAgent(ThorObjectSearchCosAgent):
    """
    This agent implements the complete

    POMDP <-> (Robot <-> World)

    In this version, COSPOMDP will use a 2D state space for
    object location. For the robot state, it will store
    both robot state at a topological graph node and
    the ground robot pose; The topological graph is automatically
    built based on sampling.
    """
    AGENT_USES_CONTROLLER=False
    def __init__(self,
                 task_config,
                 corr_specs,
                 detector_specs,
                 solver,
                 solver_args,
                 grid_map,
                 thor_agent_pose,
                 thor_prior={},
                 num_place_samples=10,
                 topo_map_degree=(3,5),
                 places_sep=4.0,
                 topo_cover_thresh=0.5,
                 local_search_type="basic",
                 local_search_params={},
                 seed=1000):
        """
        If the probability
        """

        robot_id = task_config["robot_id"]
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.search_region = search_region
        self.grid_map = grid_map
        self.reachable_positions = reachable_positions  # positions the robot can reach

        # initial robot pose
        init_robot_pose = grid_map.to_grid_pose(
            thor_agent_pose[0][0],  #x
            thor_agent_pose[0][2],  #z
            thor_agent_pose[1][1]   #yaw
        )
        pitch = thor_agent_pose[1][0]

        # Form initial topological graph for navigation.
        prior = {loc: 1e-12 for loc in search_region}
        for thor_loc in thor_prior:
            loc = grid_map.to_grid_pos(thor_loc[0], thor_loc[2])
            prior[loc] = thor_prior[thor_loc]

        self._num_place_samples = num_place_samples
        self._places_sep = places_sep
        self._topo_map_degree = topo_map_degree
        self._seed = seed
        self._topo_cover_thresh = topo_cover_thresh
        self.topo_map = _sample_topo_map(prior,
                                         reachable_positions,
                                         self._num_place_samples,
                                         degree=self._topo_map_degree,
                                         sep=self._places_sep,
                                         rnd=random.Random(self._seed),
                                         robot_pos=init_robot_pose[:2])
        init_topo_nid = self.topo_map.closest_node(*init_robot_pose[:2])
        init_robot_state = RobotStateTopo(robot_id, init_robot_pose, pitch, init_topo_nid)
        self.thor_movement_params = task_config["nav_config"]["movement_params"]

        if task_config["task_type"] == 'class':
            target_id = task_config['target']
            target_class = task_config['target']
            target = (target_id, target_class)
        else:
            target = task_config['target']  # (target_id, target_class)
            target_id = target[0]
        self.task_config = task_config
        self.target = target

        detectors, detectable_objects = self._build_detectors(detector_specs)
        corr_dists = self._build_corr_dists(corr_specs, detectable_objects)
        self.detectors = detectors
        self.corr_dists = corr_dists

        robot_trans_model = RobotTransitionTopo(robot_id, target[0],
                                                self.topo_map, self.task_config['nav_config']['h_angles'])
        reward_model = cospomdp.ObjectSearchRewardModel(
            detectors[target_id].sensor,
            (task_config["nav_config"]["goal_distance"] / grid_map.grid_size) * 0.8,  # just to make sure we are close enough
            robot_id, target_id,
            **task_config["reward_config"])
        policy_model = PolicyModelTopo(robot_trans_model, self.topo_map)
                                       # reward_model,

        prior = {grid_map.to_grid_pos(p[0], p[2]): thor_prior[p]
                 for p in thor_prior}
        self.cos_agent = cospomdp.CosAgent(self.target,
                                           init_robot_state,
                                           search_region,
                                           robot_trans_model,
                                           policy_model,
                                           corr_dists,
                                           detectors,
                                           reward_model,
                                           prior=prior)
        self._local_search_type = local_search_type
        self._local_search_params = local_search_params
        if solver == "pomdp_py.POUCT":
            self.solver = pomdp_py.POUCT(**solver_args,
                                         rollout_policy=self.cos_agent.policy_model)
        else:
            self.solver = eval(solver)(**solver_args)

        # This is used to output low-level actions that achieve
        # goals
        self._goal_handler = None

    def act(self):
        goal = self.solver.plan(self.cos_agent)
        print("Goal: {}".format(goal))
        # if goal.name == "done":
        #     from pomdp_py import TreeDebugger
        #     dd = TreeDebugger(self.cos_agent.tree)
        #     import pdb; pdb.set_trace()

        if self._goal_handler is None or goal != self._goal_handler.goal:
            # Goal is different now. We try to handle this goal
            self._goal_handler = self.handle(goal)

        action = self._goal_handler.step()
        assert isinstance(action, TOS_Action)
        return action

    def handle(self, goal):
        """Returns a handler for achieving the goal."""
        if isinstance(goal, Stay):
            return LocalSearchHandler.create(goal, self,
                                             self._local_search_type,
                                             self._local_search_params)

        elif isinstance(goal, MoveTopo):
            return MoveTopoHandler(goal, self)

        elif isinstance(goal, cospomdp.Done):
            return DoneHandler(goal, self)


    def update(self, tos_action, tos_observation):
        # Update the goal handler with low-level sensory observation
        if self._goal_handler.updates_first:
            self._goal_handler.update(tos_action, tos_observation)

        # Also update COS-POMDP with the low-level observation
        super().update(tos_action, tos_observation)

        if not self._goal_handler.updates_first:
            self._goal_handler.update(tos_action, tos_observation)

        # this shouldn't hurt, theoretically; It is necessary in order
        # to prevent replanning goals from the same, out-dated tree while
        # a goal is in execution.
        del self.cos_agent.tree # remove the search tree after planning

        # Update the topo map (resample it, because belief has changed),
        # if the belief update makes the current one undesirable
        btarget = self.belief.b(self.target_id)
        target_hist = {s.loc: btarget[s] for s in btarget}
        if self.topo_map.total_prob(target_hist) < self._topo_cover_thresh:
            self._resample_topo_map(target_hist)
            # since we updated the topological map,
            # existing search tree is invalid.
            if hasattr(self.cos_agent, "tree"):
                del self.cos_agent.tree



    def interpret_robot_obz(self, tos_observation):
        # Here, we will build a pose of format (x, y, pitch, yaw, nid)
        thor_robot_position, thor_robot_rotation = tos_observation.robot_pose
        x, y, yaw = self.grid_map.to_grid_pose(thor_robot_position['x'],
                                               thor_robot_position['z'],
                                               thor_robot_rotation['y'])
        pitch = tos_observation.horizon
        nid = self.belief.b(self.robot_id).mpe().topo_nid  # keeps the same topo node id
        return cospomdp.RobotObservation(self.robot_id,
                                         (x, y, yaw),
                                         cospomdp.RobotStatus(done=tos_observation.done),
                                         horizon=pitch,
                                         topo_nid=nid)


    def interpret_action(self, tos_action):
        # It actually doesn't matter to the CosAgent what low-level
        # action is taken by the goal handler.
        return None

    def _update_belief(self, action, observation):
        """
        Here, action, observation are already interpreted.
        This agent doesn't care about low-level action.
        """
        self.cos_agent.update(None, observation)

        # If the goal is done, we will supply the observation
        # (because COSAgent here expects observations after
        # a goal is completed. If not, then the observation
        # is not useful, and we don't update the solver
        if self._goal_handler.done:
            # will update the node id to the goal, if the handled
            # goal is movetopo
            if isinstance(self._goal_handler.goal, MoveTopo):
                srobot_old = self.belief.b(self.robot_id).mpe()
                new_nid = self.topo_map.closest_node(*srobot_old.pose[:2])
                self._update_belief_topo_nid(srobot_old, new_nid)

            self.solver.update(self.cos_agent, self._goal_handler.goal, observation)

    def _resample_topo_map(self, target_hist):
        srobot_old = self.cos_agent.belief.b(self.robot_id).mpe()
        topo_map = _sample_topo_map(target_hist,
                                    self.reachable_positions,
                                    self._num_place_samples,
                                    degree=self._topo_map_degree,
                                    sep=self._places_sep,
                                    rnd=random.Random(self._seed),
                                    robot_pos=srobot_old.pose[:2])
        self.cos_agent.transition_model.robot_trans_model.update(topo_map)
        self.cos_agent.policy_model.update(topo_map)
        self.topo_map = topo_map
        self._update_belief_topo_nid(srobot_old,
                                     topo_map.closest_node(*srobot_old.pose[:2]))

    def _update_belief_topo_nid(self, srobot_old, new_nid):
        """
        Returns a RobotStateTopo with all fields the same as srobot_old,
        except that the topo_nid is equal to the `new_nid`
        """
        srobot = RobotStateTopo(srobot_old.id,
                                srobot_old.pose,
                                srobot_old.horizon,
                                new_nid,
                                status=srobot_old.status)
        self.belief.set_b(self.robot_id,
                          pomdp_py.Histogram({srobot: 1.0}))
