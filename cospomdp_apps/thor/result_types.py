from thortils import compute_spl
from cospomdp.utils.math import ci_normal
from sciex import Result, PklResult, YamlResult
import ai2thor.util.metrics as metrics
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import os

# Note: For PklResult, it is not recommended to save the object directly
# if it is of a custom class; Use it if you only save generic python objects
# or popular objects like numpy arrays.

def baseline_name(baseline):
    mapping = {"random#gt" : 'Random',
               "greedy-nbv#vision": 'Greedy-NBV (v, acc)',
               "hierarchical#target-only#vision": 'Target-POMDP (v)',
               "hierarchical#corr#vision": 'COS-POMDP (v, acc)',
               "hierarchical#corr#gt": 'COS-POMDP (gt, acc)',
               "hierarchical#corr-learned#vision": 'COS-POMDP (v, lrn)',
               "hierarchical#corr-wrong#vision": 'COS-POMDP (v, wrg)'}
    return mapping[baseline]

class PathResult(PklResult):
    """
    Paths, includes actual path and shortest path,
    where each is a sequence robot poses tuples.
    Includes success.
    """
    def __init__(self, scene, target, shortest_path, actual_path, success,
                 rewards=[], discount_factor=None):
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
        self.rewards = rewards
        self.discount_factor = discount_factor
        super().__init__({
            "scene": self.scene,
            "target": self.target,
            "shortest_path": self.shortest_path,
            "shortest_path_distance": self.shortest_path_distance,
            "actual_path": self.actual_path,
            "actual_path_distance": self.actual_path_distance,
            "success": self.success,
            "rewards": self.rewards,
            "discount_factor": self.discount_factor
        })

    @classmethod
    def from_dict(cls, dd):
        """Not sure why but some times the pickle file is loaded as a dict"""
        return PathResult(dd['scene'],
                          dd['target'],
                          dd['shortest_path'],
                          dd['actual_path'],
                          dd['success'],
                          rewards=dd.get('rewards', []),
                          discount_factor=dd.get('discount_factor', None))

    def discounted_return(self):
        if self.discount_factor is None:
            return None

        discount = 1.0
        ret = 0.0
        for r in self.rewards:
            ret += r*discount
            discount *= self.discount_factor
        return ret

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
            episode_results = []  # results for 'episodes' (i.e. individual search trials)
            disc_returns = []   # discounted returns per trial (because each seed is a search trial for the same object)
            success_count = 0
            for seed in results[baseline]:
                path_result = results[baseline][seed]
                if type(path_result) == dict:
                    path_result = PathResult.from_dict(path_result)
                episode_results.append(path_result.to_tuple())
                disc_return = path_result.discounted_return()
                disc_returns.append(disc_return)
                if path_result.success:
                    success_count += 1

            if len(episode_results) != 0:
                spl = compute_spl(episode_results)
                rows.append([baseline, spl, success_count, len(results[baseline]), np.mean(disc_returns)])
        cls.sharedheader = ["baseline", "spl", "success", "total", "disc_return"]
        return rows

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        # Gathered results maps from global name to what is returned by gather()
        all_rows = []
        for global_name in gathered_results:
            scene_type, scene, target = global_name.split("-")
            for row in gathered_results[global_name]:
                success_rate = row[2] / max(1, row[3])
                all_rows.append([scene_type, scene, target] + row + [success_rate])
        columns = ["scene_type", "scene", "target"] + cls.sharedheader + ["success_rate"]
        df = pd.DataFrame(all_rows, columns=columns)

        ci_func = lambda x: ci_normal(x, confidence_interval=0.95)
        summary = df.groupby(['baseline'])\
                    .agg([("avg", "mean"),
                          "std",
                          ("ci95", ci_func)])
        summary_by_scene_type = df.groupby(['scene_type', 'baseline'])\
                             .agg([("avg", "mean"),
                                   "std",
                                   ("ci95", ci_func),
                                   ("count", "sum")])
        summary_by_scene = df.groupby(['scene', 'baseline'])\
                             .agg([("avg", "mean"),
                                   "std",
                                   ("ci95", ci_func)])
        summary_by_target = df.groupby(['target', 'baseline'])\
                              .agg([("avg", "mean"),
                                    "std",
                                    ("ci95", ci_func),
                                    ("count", "sum")])
        df.to_csv(os.path.join(path, "path_result.csv"))
        summary.to_csv(os.path.join(path, "path_result_summary.csv"))
        summary_by_scene_type.to_csv(os.path.join(path, "path_result_summary-by-scene-type.csv"))
        summary_by_scene.to_csv(os.path.join(path, "path_result_summary-by-scene.csv"))
        summary_by_target.to_csv(os.path.join(path, "path_result_summary-by-target.csv"))

        # First table:
        baseline_order = ["random#gt",
                          "greedy-nbv#vision",
                          "hierarchical#target-only#vision",
                          "hierarchical#corr#vision",
                          "hierarchical#corr#gt",
                          "hierarchical#corr-learned#vision",
                          "hierarchical#corr-wrong#vision"]

        # Generate tables usable in paper
        # First table: summary by scene type
        df = summary_by_scene_type
        table_rows = []
        for baseline in baseline_order:
            result_row = {('', 'Method'): baseline_name(baseline)}
            for scene_type in ["bathroom", "bedroom", "kitchen", "living_room"]:
                scene_type = scene_type.replace('_', '+')
                row = df.loc[(scene_type, baseline)]

                # spl
                spl_avg = row[('spl', 'avg')]
                spl_ci95 = row[('spl', 'ci95')]
                spl = f"{100*spl_avg:.2f} ({100*spl_ci95:.2f})"

                # discounted_return
                dr_avg = row[('disc_return', 'avg')]
                dr_ci95 = row[('disc_return', 'ci95')]
                dr = f"{dr_avg:.2f} ({dr_ci95:.2f})"
                # success rate
                sr_avg = row[('success_rate', 'avg')]
                sr = f"{100*sr_avg:.2f}"

                result_row[(scene_type, 'SPL (%)')] = spl
                result_row[(scene_type, 'DR')] = dr
                result_row[(scene_type, 'SR (%)')] = sr
            table_rows.append(result_row)
        df_main = pd.DataFrame(table_rows)
        df_main.columns = pd.MultiIndex.from_tuples(df_main.transpose().index, names=['scene_type', 'metric'])
        df_main.index = df_main.loc[:, ('', 'Method')]
        df_main = df_main.iloc[:, 1:]
        print(df_main.to_latex())

        # Second table: summary by objects
        import sys
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(ABS_PATH, "../../experiments/thor"))
        from experiment_thor import OBJECT_CLASSES
        df = summary_by_target
        scene_target_data = []
        for scene_type in sorted(OBJECT_CLASSES):
            for target in OBJECT_CLASSES[scene_type]['target']:
                result_row = {
                    ('', 'Room Type') : scene_type,
                    ('', 'Target Class') : target
                }
                for baseline in ["greedy-nbv#vision", "hierarchical#target-only#vision", "hierarchical#corr#vision"]:
                    row = df.loc[(target, baseline)]
                    # spl
                    spl_avg = row[('spl', 'avg')]
                    spl_ci95 = row[('spl', 'ci95')]
                    spl = f"{100*spl_avg:.2f} ({100*spl_ci95:.2f})"

                    # discounted_return
                    dr_avg = row[('disc_return', 'avg')]
                    dr_ci95 = row[('disc_return', 'ci95')]
                    dr = f"{dr_avg:.2f} ({dr_ci95:.2f})"
                    # success rate
                    sr_avg = row[('success_rate', 'avg')]
                    sr = f"{100*sr_avg:.2f}"

                    result_row[(baseline_name(baseline), 'SPL (%)')] = spl
                    result_row[(baseline_name(baseline), 'DR')] = dr
                    result_row[(baseline_name(baseline), 'SR (%)')] = sr
                scene_target_data.append(result_row)
        df_target = pd.DataFrame(scene_target_data)
        df_target.columns = pd.MultiIndex.from_tuples(df_target.transpose().index, names=['baseline', 'metric'])
        df_target.index = pd.MultiIndex.from_tuples(zip(df_target.loc[:, ('', 'Room Type')], df_target.loc[:, ('', 'Target Class')]))
        df_target = df_target.iloc[:, 2:]
        print(df_target.to_latex(multirow=True))



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

    @classmethod
    def collect(cls, path):
        # For efficiency
        pass

    def discounted_return(self):
        discount = 1.0
        ret = 0.0
        for step in self.history:
            ret += step['reward']*discount
            discount *= self.discount_factor
        return ret
