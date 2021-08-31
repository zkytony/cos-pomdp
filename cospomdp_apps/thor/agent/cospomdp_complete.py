import pomdp_py
from collections import deque
from thortils.navigation import find_navigation_plan
from ..common import ThorAgent, TOS_Action
from .cospomdp_basic import GridMapSearchRegion, ThorObjectSearchCosAgent
from .components.action import (grid_navigation_actions,
                                from_grid_action_to_thor_action_delta)
from .components.state import CosRobotState
from .components.topo_map import TopoNode, TopoMap, TopoEdge
from ..constants import GOAL_DISTANCE
from cospomdp.utils.math import euclidean_dist
from cospomdp.models.agent import CosAgent
from cospomdp.models.reward_model import ObjectSearchRewardModel

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


def _sample_places(target_hist,
                   reachable_positions,
                   num_places,
                   degree=3,
                   sep=4.0):
    """Given a search region, a distribution over target locations in the
    search region, return a list of `num_places` of locations within
    reachable_positions.

    The algorithm works by first converting the target_hist,
    which is a distribution over the search region, to a distribution
    over the robot's reachable positions.

    This is done by, for each location in the search region, find
    a closest reachable position; Then the probability at a reachable
    position is the sum of those search region locations mapped to it.

    Then, simply sample reachable positions based on this distribution.

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
    hist = pomdp_py.Histogram(reachable_pos_dist)

    places = set()
    for i in range(num_places):
        pos = hist.random()
        places.add(pos)

    # Create nodes
    pos_to_nid = {}
    nodes = {}
    for i, pos in enumerate(places):
        topo_node = TopoNode(i, pos)
        nodes[i] = topo_node
        pos_to_nid[pos] = i

    # Now, we need to connect the places to form a graph.
    edges = {}
    for nid in nodes:
        topo_node = nodes[i]
        pos = topo_node.pos

        neighbors = set()
        candidates = set(places) - {pos}
        while len(candidates) > 0:
            closest_pos = min(candidates,
                              key=lambda c: euclidean_dist(pos, c))
            if euclidean_dist(closest_pos, pos) >= sep:
                # add neighbor
                nb_nid = pos_to_nid[closest_pos]
                neighbors.add(nb_nid)
                if len(neighbors) >= degree:
                    break
            candidates -= {closest_pos}

        for nb_nid in neighbors:
            # Find a path between
            path = _shortest_path(reachable_positions,
                                  nodes[nb_nid].pos,
                                  topo_node.pos)
            if path is None:
                # Skip this edge because we cannot find path
                continue
            eid = len(edges)
            edges[eid] = TopoEdge(eid,
                                  topo_node,
                                  nodes[nb_nid],
                                  path)
    return TopoMap(edges)


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
                 grid_map,
                 thor_agent_pose,
                 thor_prior={},
                 num_place_samples=10,
                 topo_map_degree=3,
                 places_sep=4.0):

        robot_id = task_config["robot_id"]
        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations
        self.search_region = search_region
        self.grid_map = grid_map

        # Form initial topological graph for navigation.
        prior = {loc: 1e-12 for loc in search_region}
        for thor_loc in thor_prior:
            loc = grid_map.to_grid_pos(thor_loc[0], thor_loc[2])
            prior[loc] = thor_prior[thor_loc]

        movement_params = task_config["nav_config"]["movement_params"]
        init_robot_pose = grid_map.to_grid_pose(
            thor_agent_pose[0][0],  #x
            thor_agent_pose[0][2],  #z
            thor_agent_pose[1][1]   #yaw
        )
        pitch = thor_agent_pose[1][0]
        self.lll = _sample_places(prior,
                                  reachable_positions,
                                  num_place_samples,
                                  degree=topo_map_degree,
                                  sep=places_sep)

        init_robot_state = CosRobotState(robot_id, init_robot_pose, pitch, None)

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

        reward_model = ObjectSearchRewardModel(
            detectors[target_id].sensor,
            task_config["nav_config"]["goal_distance"] / grid_map.grid_size,
            robot_id, target_id)

        self.cos_agent = CosAgent(self.target,
                                  init_robot_state,
                                  search_region,
                                  None,  # TODO
                                  None,  # TODO
                                  corr_dists,
                                  detectors,
                                  reward_model)



    def act(self):
        return TOS_Action("Pass", {})
