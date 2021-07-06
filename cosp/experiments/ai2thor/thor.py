import pomdp_py
import ai2thor
import ai2thor.util.metrics as metrics
from cosp import TaskEnv
from sciex import Result, PklResult

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

class ThorObjectSearch(ThorEnv):
    """
    This represents the environment of running a single object search task.
    """
    def __init__(self, controller,
                 target_type, target,
                 goal_distance=1.0):
        """
        If target_type is "class", then target is an object type.
        If target_type is "object", then target is an object ID.
        """
        if self.target_type not in {"class", "object"}:
            raise ValueError("Invalid target type: {}".format(self.target_type))
        super().__init__(controller)
        self.target = target
        self.target_type = target_type
        self.goal_distance = goal_distance


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
        if self.target_type == "class":
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


class SingleSPLResult(PklResult):
    def __init__(self, shortest_path, actual_path, success):
        shortest_path_distance = metrics.path_distance(shortest_path)
        actual_path_distance = metrics.path_distance(actual_path)
        super().__init__({
            "shortest_path": shortest_path,
            "shortest_path_distance": shortest_path_distance,
            "actual_path": actual_path,
            "actual_path_distance": shortest_path_distance
        })
    @classmethod
    def FILENAME(cls):
        return "paths.pkl"


class HistorySPLResult(PklResult):
    def __init__(self, history):
        """
        History is a list of (s, a, o, r) tuples
        """
        super().__init__(history)
    @classmethod
    def FILENAME(cls):
        return "history.pkl"


class ThorAgent(Agent):
    pass
