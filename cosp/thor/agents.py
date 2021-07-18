from ai2thor.util.metrics import (get_shortest_path_to_object,
                                  get_shortest_path_to_object_type)

from thortils import (thor_agent_pose,
                      thor_closest_object_of_type,
                      thor_object_pose,
                      thor_reachable_positions)
from thortils.navigation import find_navigation_plan, get_navigation_actions
from .utils import plot_path, plt
from ..framework import Agent


class ThorAgent(Agent):
    def __init__(self, movement_params):
        self.movement_params = movement_params
        # The navigation_actions here is a list of tuples
        # (movement_str, (forward, h_angle, v_angle))
        self.navigation_actions =\
            get_navigation_actions(self.movement_params)

    def act(self):
        pass

    def update(self, observation, reward):
        pass


class ThorObjectSearchAgent(ThorAgent):
    AGENT_USES_CONTROLLER = False


class ThorObjectSearchOptimalAgent(ThorObjectSearchAgent):
    """
    The optimal agent uses ai2thor's shortest path method
    to retrieve a path, and then follows this path by taking
    appropriate actions.
    """

    # Because this agent is not realistic, we permit it to have
    # access to the controller.
    AGENT_USES_CONTROLLER = True

    def __init__(self, controller, target, task_type,
                 movement_params):
        super().__init__(movement_params)
        self.controller = controller
        self.target = target
        self.task_type = task_type

        if task_type == "class":
            obj = thor_closest_object_of_type(controller, self.target)
            target_position = (obj["position"]["x"],
                               obj["position"]["y"],
                               obj["position"]["z"])
        else:
            target_position = thor_object_pose(controller,
                                               self.target, as_tuple=True)

        start_pose = thor_agent_pose(self.controller, as_tuple=True)
        goal_pose = (target_position, start_pose[1])
        plan = find_navigation_plan(start_pose, goal_pose,
                                    self.navigation_actions,
                                    thor_reachable_positions(controller))
        import pdb; pdb.set_trace()



    def update(self, action, observation):
        pass
