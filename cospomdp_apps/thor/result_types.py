from thortils import compute_spl
from cospomdp.utils.math import ci_normal
from sciex import Result, PklResult, YamlResult
# import ai2thor.util.metrics as metrics
import pandas as pd
import seaborn as sns
import pickle
import os

# Note: For PklResult, it is not recommended to save the object directly
# if it is of a custom class; Use it if you only save generic python objects
# or popular objects like numpy arrays.

class PathResult(PklResult):
    """
    Paths, includes actual path and shortest path,
    where each is a sequence robot poses tuples.
    Includes success.
    """
    def __init__(self, scene, target, shortest_path, actual_path, success):
        """
        Args:
            scene (str): Scene of the seach trial (floor plan)
            target (str): target (object ID or class)
            shortest_path (list): List of robot pose tuples (the best)
            actual_path (list): List of robot pose tuples (the actual)
            success (bool): Success/fail of the task
        """
        self.shortest_path = shortest_path
        self.actual_path = actual_path
        self.success = success
        self.shortest_path_distance = metrics.path_distance(shortest_path)
        self.actual_path_distance = metrics.path_distance(actual_path)
        self.scene = scene
        self.target = target
        super().__init__({
            "scene": self.scene,
            "target": self.target,
            "shortest_path": self.shortest_path,
            "shortest_path_distance": self.shortest_path_distance,
            "actual_path": self.actual_path,
            "actual_path_distance": self.actual_path_distance,
            "success": self.success
        })

    @classmethod
    def from_dict(cls, dd):
        """Not sure why but some times the pickle file is loaded as a dict"""
        return PathResult(dd['scene'],
                          dd['target'],
                          dd['shortest_path'],
                          dd['actual_path'],
                          dd['success'])

    @classmethod
    def FILENAME(cls):
        return "paths.pkl"

    def to_tuple(self):
        """Returns (shortest_path_distance, actual_path_distance, success) tuples"""
        return (self.shortest_path_distance,
                self.actual_path_distance,
                self.success)

    @classmethod
    def collect(cls, path):
        # This should be standardized.
        with open(path, 'rb') as f:
            path_result = pickle.load(f)
        return path_result


    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        rows = []
        for baseline in results:
            episode_results = []
            success_count = 0
            for seed in results[baseline]:
                path_result = results[baseline][seed]
                if type(path_result) == dict:
                    path_result = PathResult.from_dict(path_result)
                episode_results.append(path_result.to_tuple())
                if path_result.success:
                    success_count += 1
            if len(episode_results) != 0:
                spl = compute_spl(episode_results)
                rows.append([baseline, spl, success_count, len(results[baseline])])
        cls.sharedheader = ["baseline", "spl", "success", "total"]
        return rows

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        # Gathered results maps from global name to what is returned by gather()
        all_rows = []
        for global_name in gathered_results:
            scene, target, other = global_name.split("-")
            for row in gathered_results[global_name]:
                success_rate = row[2] / max(1, row[3])
                all_rows.append([scene, target, other] + row + [success_rate])
        columns = ["scene", "target_class", "other_class"] + cls.sharedheader + ["success_rate"]
        df = pd.DataFrame(all_rows, columns=columns)

        ci_func = lambda x: ci_normal(x, confidence_interval=0.95)
        summary = df.groupby(['baseline'])\
                    .agg([("avg", "mean"),
                          "std",
                          ("ci95", ci_func)])
        summary_by_scene = df.groupby(['scene', 'baseline'])\
                             .agg([("avg", "mean"),
                                   "std",
                                   ("ci95", ci_func)])
        summary_by_target = df.groupby(['target_class', 'baseline'])\
                              .agg([("avg", "mean"),
                                    "std",
                                    ("ci95", ci_func)])
        df.to_csv(os.path.join(path, "path_result.csv"))
        summary.to_csv(os.path.join(path, "path_result_summary.csv"))
        summary_by_scene.to_csv(os.path.join(path, "path_result_summary-by-scene.csv"))
        summary_by_target.to_csv(os.path.join(path, "path_result_summary-by-target.csv"))


class HistoryResult(YamlResult):
    def __init__(self, history, discount_factor):
        """history is a list of {s, a, o, r} dictionaries
        Assume that each value has been formatted for readability and parsing."""
        self.discount_factor = discount_factor
        self.history = history
        super().__init__({"history": history, "discount_factor": discount_factor})

    @classmethod
    def FILENAME(cls):
        return "history.yaml"

    def discounted_return(self):
        discount = 1.0
        ret = 0.0
        for step in self.history:
            sp, a, o, r = step
            ret += r*discount
            discount *= self.discount_factor
        return ret
