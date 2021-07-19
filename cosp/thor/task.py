import pomdp_py
import ai2thor
import ai2thor.util.metrics as metrics

from . import utils
from ..framework import TaskEnv
from ..utils.math import euclidean_dist


class ThorEnv(TaskEnv):
    def __init__(self, controller):
        self.controller = controller
        self._history = []  # stores the (s, a, o, r) tuples so far

    @property
    def init_state(self):
        return self._history[0][0]

    def execute(self, action):
        event = self.controller.step(action.name, **action.params)

    def done(self, action):
        event = self.controller.step(action.name, **action.params)



# ------------- Object Search ------------- #
class ThorObjectSearch(ThorEnv):
    """
    This represents the environment of running a single object search task.
    """
    def __init__(self, controller, task_config):
        """
        If task_type is "class", then target is an object type.
        If task_type is "object", then target is an object ID.
        """
        task_type = task_config["task_type"]
        target = task_config["target"]
        if task_type not in {"class", "object"}:
            raise ValueError("Invalid target type: {}".format(task_type))
        super().__init__(controller)
        self.target = target
        self.task_type = task_type
        self.goal_distance = task_config["goal_distance"]


    def compute_results(self):
        """
        We will compute:
        1. Discounted cumulative reward
           Will save the entire trace of history.

        2. SPL. Even though our problem involves open/close,
           the optimal path should be just the navigation length,
           because the agent just needs to navigate to the container, open it
           and then perhaps look down, which doesn't change the path length.
           This metric alone won't tell the full story. Because it obscures the
           cost of actions that don't change the robot's location. So a combination with
           discounted reward is better.

           Because SPL is a metric over all trials, we will return the
           result for individual trials, namely, the path length, shortest path length,
           and success
        """
        if self.task_type == "class":
            shortest_path = metrics.get_shortest_path_to_object_type(
                self.controller, self.target,
                self.init_state.position, init_rotation=self.init_state.rotation)
        else:
            shortest_path = metrics.get_shortest_path_to_object(
                self.controller, self.target,
                self.init_state.position, init_rotation=self.init_state.rotation)
        actual_path = self.get_current_path()
        success = self.done()
        return [SingleSPLResult(shortest_path, actual_path, success),
                HistoryResult(self._history)]

    def get_current_path(self):
        """Get the path currently in history.  As with ai2thor, the path is a list of
        dicts where each represents position/rotation at a point.
        """
        raise NotImplementedError

    def done(self):
        event = self.controller.step(action="Pass")
        visible_objects = utils.thor_visible_objects(event)
        agent_position = utils.thor_agent_position(event)
        for obj in visible_objects:
            if self.task_type == "class":
                if obj["objectType"] == self.target:
                    if euclidean_dist(obj["position"], agent_position) <= self.goal_distance:
                        return True
            else:
                if obj["objectId"] == self.target:
                    if euclidean_dist(obj["position"], agent_position) <= self.goal_distance:
                        return True
        return False
