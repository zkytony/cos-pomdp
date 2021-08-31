import pomdp_py
import time
import random
from collections import deque
from thortils.navigation import find_navigation_plan, get_navigation_actions
from cospomdp.utils.math import euclidean_dist, normalize
from cospomdp.models.agent import CosAgent
from cospomdp.models.reward_model import ObjectSearchRewardModel
from ..common import ThorAgent, TOS_Action
from .cospomdp_basic import GridMapSearchRegion, ThorObjectSearchCosAgent
from .components.action import (grid_navigation_actions,
                                from_grid_action_to_thor_action_delta,
                                from_grid_action_to_thor_action_params,
                                from_thor_delta_to_thor_action_params,
                                Move, MoveTopo)
from cospomdp.domain.action import Done
from .components.state import RobotStateTopo
from .components.topo_map import TopoNode, TopoMap, TopoEdge
from .components.transition_model import RobotTransitionTopo
from .components.policy_model import PolicyModelTopo
from ..constants import GOAL_DISTANCE


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
                     degree=3,
                     sep=4.0,
                     rnd=random):
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
        degrees (int): controls the number of maximum neighbors per place.
        sep (float): minimum distance between two places (grid cells)

    Returns:
        TopologicalMap.

    """
    mapping = {}
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
        topo_node = TopoNode(i, pos, hist[pos])
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
        degree_needed = degree - len(neighbors)
        new_neighbors = list(sorted(candidates, key=lambda pos: euclidean_dist(pos, nodes[nid].pos)))[:degree_needed]
        for nbpos in new_neighbors:
            nbnid = pos_to_nid[nbpos]
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
    topo_map = TopoMap(edges)
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
                 topo_map_degree=3,
                 places_sep=4.0,
                 seed=1000):

        robot_id = task_config["robot_id"]
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.search_region = search_region
        self.grid_map = grid_map
        self.reachable_positions = reachable_positions  # positions the robot can reach

        # Form initial topological graph for navigation.
        prior = {loc: 1e-12 for loc in search_region}
        for thor_loc in thor_prior:
            loc = grid_map.to_grid_pos(thor_loc[0], thor_loc[2])
            prior[loc] = thor_prior[thor_loc]

        self.thor_movement_params = task_config["nav_config"]["movement_params"]
        init_robot_pose = grid_map.to_grid_pose(
            thor_agent_pose[0][0],  #x
            thor_agent_pose[0][2],  #z
            thor_agent_pose[1][1]   #yaw
        )
        pitch = thor_agent_pose[1][0]
        self.topo_map = _sample_topo_map(prior,
                                         reachable_positions,
                                         num_place_samples,
                                         degree=topo_map_degree,
                                         sep=places_sep,
                                         rnd=random.Random(seed))

        init_topo_nid = self.topo_map.closest_node(*init_robot_pose[:2])
        init_robot_state = RobotStateTopo(robot_id, init_robot_pose, pitch, init_topo_nid)

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

        robot_trans_model = RobotTransitionTopo(robot_id, target[0],
                                                self.topo_map, self.task_config['nav_config']['h_angles'])
        reward_model = ObjectSearchRewardModel(
            detectors[target_id].sensor,
            task_config["nav_config"]["goal_distance"] / grid_map.grid_size,
            robot_id, target_id,
            **task_config["reward_config"])
        policy_model = PolicyModelTopo(robot_trans_model,
                                       reward_model,
                                       self.topo_map)

        self.cos_agent = CosAgent(self.target,
                                  init_robot_state,
                                  search_region,
                                  robot_trans_model,
                                  policy_model,
                                  corr_dists,
                                  detectors,
                                  reward_model)
        if solver == "pomdp_py.POUCT":
            self.solver = pomdp_py.POUCT(**solver_args,
                                         rollout_policy=self.cos_agent.policy_model)
        else:
            self.solver = eval(solver)(**solver_args)

        # This is used to output low-level actions that achieve
        # goals
        self._goal_handler = None

    def act(self):
        if self._goal_handler is None or self._goal_handler.done:
            goal = self.solver.plan(self.cos_agent)
            self._goal_handler = self.handle(goal)

        # Low-level action
        action = self._goal_handler.step()
        assert isinstance(action, TOS_Action)
        return action

    def update(self, tos_action, tos_observation):
        # Update the goal handler with low-level sensory observation
        self._goal_handler.update(tos_action, tos_observation)

        # Interpret the observation and update the COS-POMDP agent's belief.
        print("TODO")
        thor_robot_pose = tos_observation.robot_pose
        thor_camera_horizon = tos_observation.horizon  # i.e. pitch
        robot_pose = self.grid_map.to_grid_pose(thor_robot_pose[0]['x'],
                                                thor_robot_pose[0]['z'],
                                                thor_robot_pose[1]['y'])
        # TODO: properly set status - right now there is only 'done' and it
        # doesn't affect behavior if this is always false because task success
        # depends on taking the done action, not the done status.
        status = RobotStatus()
        robotobz = RobotObservation(self.robot_id, robot_pose, status)
        observation = CosObservation(robotobz, objobzs)





    def handle(self, goal):
        """Returns a handler for achieving the goal."""
        if isinstance(goal, MoveTopo):
            return MoveTopoHandler(goal, self)

        elif isinstance(goal, Done):
            pass

# -------- Goal Handlers -------- #
class GoalHandler:
    def __init__(self, goal, agent):
        """
        agent: ThorObjectSearchCompleteCosAgent
        """
        pass

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
        return "{}({})".format(self.__class__.__name__, str(self))



class MoveTopoHandler(GoalHandler):
    """Deals with navigating along an edge on the topological map."""
    def __init__(self, goal, agent):
        assert isinstance(goal, MoveTopo)
        # Plans a sequence of actions to go from where
        # the robot is currently to the dst node.
        robot_pose = agent.belief.random().s(agent.robot_id).pose
        closest_nid = agent.topo_map.closest_node(*robot_pose[:2])
        assert closest_nid == goal.src_nid,\
            "Not expecting agent, which is closest to node {}, to take {}"\
            .format(closest_nid, goal)

        # Preparing the inputs for find_navigation_plan in thortils
        thor_rx, thor_rz, thor_rth = agent.grid_map.to_thor_pose(*robot_pose)
        thor_start_position = (thor_rx, 0, thor_rz)
        thor_start_rotation = (0, thor_rth, 0)
        _goal_pos = agent.topo_map.nodes[goal.dst_nid].pos
        thor_gx, thor_gz = agent.grid_map.to_thor_pos(*_goal_pos)
        thor_goal_position = (thor_gx, 0, thor_gz)
        thor_goal_rotation = (0, 0, 0)  # we don't care about rotation here
        thor_reachable_positions = [agent.grid_map.to_thor_pos(*p)
                                    for p in agent.reachable_positions]
        navigation_actions = get_navigation_actions(agent.thor_movement_params)
        plan, _ = find_navigation_plan((thor_start_position, thor_start_rotation),
                                       (thor_goal_position, thor_goal_rotation),
                                       navigation_actions,
                                       thor_reachable_positions,
                                       grid_size=agent.grid_map.grid_size,
                                       diagonal_ok=agent.task_config["nav_config"]["diagonal_ok"],
                                       debug=True)
        self._plan = plan
        self._index = 0


    def step(self):
        action_name, action_delta = self._plan[self._index]["action"]
        params = from_thor_delta_to_thor_action_params(action_name, action_delta)
        return TOS_Action(action_name, params)

    def update(self, tos_action, tos_observation):
        # Check if the robot pose is expected
        thor_rx = tos_observation.robot_pose[0]['x']
        thor_rz = tos_observation.robot_pose[0]['z']
        thor_rth = tos_observation.robot_pose[1]['y']
        actual = (thor_rx, thor_rz, thor_rth)
        expected_thor_rx, expected_thor_rz, _, expected_thor_rth = self._plan[self._index]['next_pose']
        expected = (expected_thor_rx, expected_thor_rz, expected_thor_rth)
        if expected != actual:
            print("Warning: after taking {}, the robot pose is unexpected.\n"
                  "The expected pose is: {}; The actual pose is: {}"\
                  .format(tos_action, expected, actual))
        self._index += 1

    def done(self):
        return self._index >= len(self._plan)
