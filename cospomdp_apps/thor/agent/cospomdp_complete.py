import pomdp_py
from ..common import ThorAgent, TOS_Action
from .cospomdp_basic import GridMapSearchRegion, ThorObjectSearchCosAgent
from .components.action import navigation_actions
from .components.state import CosRobotState
from cospomdp.models.agent import CosAgent
from cospomdp.models.reward_model import ObjectSearchRewardModel

def _sample_places(target_hist,
                   reachable_positions,
                   navigation_actions,
                   current_robot_pose,
                   num_samples):
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
                 num_place_samples=10):

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
        nav_actions = navigation_actions(movement_params, grid_map.grid_size)
        init_robot_pose = grid_map.to_grid_pose(
            thor_agent_pose[0][0],  #x
            thor_agent_pose[0][2],  #z
            thor_agent_pose[1][1]   #yaw
        )
        pitch = thor_agent_pose[1][0]
        self.lll = _sample_places(prior,
                                  reachable_positions,
                                  nav_actions,
                                  init_robot_pose,
                                  num_place_samples)

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
