import pomdp_py
from ..common import ThorAgent
from .cospomdp_basic import GridMapSearchRegion
from cospomdp_apps.basic.action import Move2D

def _sample_places(target_hist,
                   reachable_positions,
                   navigation_actions,
                   current_robot_pose,
                   num_samples,
                   distance):
    """Given a search region, a distribution over target locations
    in the search region, return a list of `num_places` of
    locations within reachable_positions.

    The algorithm works by first sampling locations from the search region based
    on the target_hist, then expanding a tree from where the robot is; For each
    new node, check if it is 'covering' any sampled location; If so, consider it
    a node on the topological graph. New nodes are searched for until all sampled
    locations are covered.

    The paths to all topological graph nodes are also returned, to save
    time for later planning.

    Note that search region, reachable_positions, navigation actions,
    current robot pose, target_hist, are all in 2D. All locations are
    on the grid map.

    Args:
        target_hist (dict): maps from location to probability
        reachable_positions (list of tuples)
        navigation_actions
        current_robot_pose
        num_places (int): number of places to sample
        min_sep (float): The minimum separation between two sampled target locations.
        distance (float): Distance between topo node (i.e. place) and sampled target location.

    Returns:
        TopologicalMap.
    """
    hist = pomdp_py.Histogram(target_hist)
    locations = []
    for i in range(num_samples):
        loc = hist.random()
        locations.append(loc)

    return locations


class ThorObjectSearchCompleteCosAgent(ThorAgent):
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
                 grid_size,
                 grid_map,
                 thor_agent_pose,
                 thor_prior={}):

        search_region = GridMapSearchRegion(grid_map)
        reachable_positions = grid_map.free_locations

        # Form initial topological graph for navigation.
        prior = {loc: 1e-12 for loc in search_region}
        for thor_loc in thor_prior:
            loc = grid_map.to_grid_pos(thor_loc[0], thor_loc[2])
            prior[loc] = thor_prior[thor_loc]



        movement_params = task_config["nav_config"]["movement_params"]
        for name in movement_params:
            action = Move2D()

        _sample_places(prior,
                       reachable_positions,
                       navigation_actions,
                       current_robot_pose,
                       num_samples,
                       distance)
