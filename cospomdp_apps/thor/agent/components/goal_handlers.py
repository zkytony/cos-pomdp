from thortils.navigation import find_navigation_plan, get_navigation_actions

import cospomdp
from cospomdp_apps.thor.common import TOS_Action
from cospomdp.utils.math import approx_equal
from .action import (MoveTopo,
                     from_thor_delta_to_thor_action_params,)


# -------- Goal Handlers -------- #
class GoalHandler:
    def __init__(self, goal, agent):
        """
        agent: ThorObjectSearchCompleteCosAgent
        """
        self.goal = goal

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
        return "{}({})".format(self.__class__.__name__, self)



class MoveTopoHandler(GoalHandler):
    """Deals with navigating along an edge on the topological map."""
    def __init__(self, goal, agent):
        assert isinstance(goal, MoveTopo)
        super().__init__(goal, agent)
        # Plans a sequence of actions to go from where
        # the robot is currently to the dst node.
        robot_pose = agent.belief.random().s(agent.robot_id).pose

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
                                       angle_tolerance=360,
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
        if not approx_equal(expected, actual, epsilon=1e-4):
            print("Warning: after taking {}, the robot pose is unexpected.\n"
                  "The expected pose is: {}; The actual pose is: {}"\
                  .format(tos_action, expected, actual))
        self._index += 1

    @property
    def done(self):
        return self._index >= len(self._plan)


class DoneHandler(GoalHandler):
    def __init__(self, goal, agent):
        super().__init__(goal, agent)

    def step(self):
        return TOS_Action(cospomdp.Done().name, {})

    def update(self, tos_action, tos_observation):
        pass

    @property
    def done(self):
        return True
