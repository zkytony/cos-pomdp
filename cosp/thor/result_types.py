from sciex import Result, PklResult, YamlResult
import ai2thor.util.metrics as metrics

# Note: For PklResult, it is not recommended to save the object directly
# if it is of a custom class; Use it if you only save generic python objects
# or popular objects like numpy arrays.

class PathResult(PklResult):
    """
    Paths, includes actual path and shortest path,
    where each is a sequence robot poses tuples.
    Includes success.
    """
    def __init__(self, shortest_path, actual_path, success):
        """
        Args:
            shortest_path (list): List of robot pose tuples (the best)
            actual_path (list): List of robot pose tuples (the actual)
            success (bool): Success/fail of the task
        """
        self.shortest_path = shortest_path
        self.actual_path = actual_path
        self.success = success
        self.shortest_path_distance = metrics.path_distance(shortest_path)
        self.actual_path_distance = metrics.path_distance(actual_path)
        super().__init__({
            "shortest_path": self.shortest_path,
            "shortest_path_distance": self.shortest_path_distance,
            "actual_path": self.actual_path,
            "actual_path_distance": self.actual_path_distance,
            "success": self.success
        })

    @classmethod
    def FILENAME(cls):
        return "paths.yaml"

    def to_tuple(self):
        """Returns (shortest_path_distance, actual_path_distance, success) tuples"""
        return (self.shortest_path_distance,
                self.actual_path_distance,
                self.success)


class HistoryResult(YamlResult):
    def __init__(self, history):
        """history is a list of {s, a, o, r} dictionaries
        Assume that each value has been formatted for readability and parsing."""
        super().__init__(history)
    @classmethod
    def FILENAME(cls):
        return "history.pkl"
