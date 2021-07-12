import ai2thor
from ai2thor.util.metrics import get_shortest_path_to_object,\
    get_shortest_path_to_object_type

from .utils import thor_agent_pose
from ..framework import Agent

class ThorAgent(Agent):
    def act(self):
        pass

    def update(self, observation, reward):
        pass

class ThorObjectSearchAgent(ThorAgent):
    AGENT_USES_CONTROLLER = False


class ThorObjectSearchOptimalAgent(ThorObjectSearchAgent):
    """The purpose of the optimal agent is to give the best path towards retrieving
    an object. That is the actions that follow shortest path to the actual
    target location. If the target is inside a container, then the agent
    should first navigate in front of that container, within interaction distance,
    then open the container, then retrieve the object. That is the best you can do,
    although no partially observable agent will ever be able to do that.

    This agent is used to test the set up and provide a way to compute
    evaluation metric, if needed.

    The path can be computed on-the-fly, as soon as a target is provided.

    We will actually consider CONTINUOUS underlying state space. But the
    actions are discrete. How to solve this path search problem?

    ... Well, turns out we can just use ai2thor's built-in method.
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
        print(position, rotation)
        print(path)
