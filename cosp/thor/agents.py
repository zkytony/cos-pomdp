from ai2thor.util.metrics import (get_shortest_path_to_object,
                                  get_shortest_path_to_object_type)

from thortils import thor_agent_pose
from .utils import plot_path, plt
from ..framework import Agent


class ThorAgent(Agent):
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

    def __init__(self, controller, target, task_type):
        self.controller = controller
        self.target = target
        self.task_type = task_type

        # Compute the shortest path
        position, rotation = thor_agent_pose(self.controller)
        if task_type == "class":
            path = get_shortest_path_to_object_type(controller,
                                                    self.target,
                                                    initial_position=position,
                                                    initial_rotation=rotation)
        else:
            path = get_shortest_path_to_object(self.controller,
                                               target,
                                               initial_position=position,
                                               initial_rotation=rotation)
        self.path = path
        plot_path(self.path, self.controller)
        plt.show()
        self._actions_pending = []
        self._index = 0


    def act(self):
        if len(self._actions_pending) == 0:
            next_position = self.path[self._index]
            import pdb; pdb.set_trace()




        pass



    def update(self, action, observation):
        pass
