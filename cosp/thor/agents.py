from . import ThorAgent


class ThorObjectSearchAgent(ThorAgent):
    pass


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
    """
    def __init__(self):
        pass
