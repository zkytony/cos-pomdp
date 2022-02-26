from thortils import compute_spl
from cospomdp.utils.math import ci_normal, test_significance_pairwise, euclidean_dist
from cospomdp_apps.thor.constants import (
    SCENE_TYPES,
    MOVE_STEP_SIZE,
    H_ROTATION,
    MAX_STEPS
)
from sciex import Result, PklResult, YamlResult
import ai2thor.util.metrics as metrics
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import sys
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

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

# True positive rate of object detection, BELOW (not including) which the class is
# considered to be hard to detect.
HARD_TO_DETECT_TP = 0.4
# Print string representation of statistical significance (e.g. *, **, ns)
# instead of printing the p value
SIGSTR = False
# Statistical significance testing method
SIGMETHOD = "wilcoxon"

# Used to estimate path time
LINEAR_SPEED=0.5  # m/s; slightly faster than roomba (0.3m/s)
ANGULAR_SPEED=45  # degree/s; typical speed (my experience).
## Necessary to estimate room sizes
sys.path.insert(0, os.path.join(ABS_PATH, "../../experiments/thor"))
from collect_scene_sizes import collect_val_scene_sizes
print("Collecting scene sizes...")
SCENE_SIZES=collect_val_scene_sizes()
# Assumes it takes about 5 seconds to search within an area of size 1m^2; unit: second per meter squared
EST_COVER_RATE=5


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
        self.shortest_path_distance = metrics.path_distance(shortest_path) if shortest_path is not None else None
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

    def estimate_time(self):
        """If the search is successful, based purely on the search trajectory,
        estimate the time taken to execute this trajectory using a
        typical linear and angular velocity. If the search is
        unsuccessful, then we set the estimated time to be EST_COVER_RATE(s/m^2)*Area.

        Note that we cannot ignore the time of these failed
        trajectories because otherwise we are not comparing the times
        from the same runs. Also, when failure happens, we cannot make
        use of the actual path because it might be short, but the
        robot had made a mistake at Done; Technically, it would have
        to start searching again.
        """
        if not self.success:
            # failure
            width, length = SCENE_SIZES[self.scene]['dim']
            return EST_COVER_RATE*(width*length)
        else:
            traj_time = 0
            for i in range(1, len(self.actual_path)):
                # a point is (x, y, z); we can obtain the
                # linear difference by subtracting the x, z
                # coordinates. If they are the same, then
                # a rotation took place. We use the rotation
                # size defined in constants.
                xcur, zcur = self.actual_path[i]['x'], self.actual_path[i]['z']
                xprev, zprev = self.actual_path[i-1]['x'], self.actual_path[i-1]['z']
                if xcur == xprev and zcur == zprev:
                    # rotation action.
                    traj_time += H_ROTATION / ANGULAR_SPEED
                else:
                    distdiff = euclidean_dist((xcur, zcur), (xprev, zprev))
                    traj_time += distdiff / LINEAR_SPEED
            return traj_time

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
            disc_returns = []     # discounted returns per trial (because each seed is a search trial for the same object)
            success_count = 0
            time_estimates = []   # actual path distances for successful trials (added for camera-ready)
            for seed in results[baseline]:
                path_result = results[baseline][seed]
                if type(path_result) == dict:
                    path_result = PathResult.from_dict(path_result)
                episode_results.append(path_result.to_tuple())
                disc_return = path_result.discounted_return()
                disc_returns.append(disc_return)
                if path_result.success:
                    success_count += 1
                time_estimates.append(path_result.estimate_time())

            if len(episode_results) != 0 and all([None not in res for res in episode_results]):
                spl = compute_spl(episode_results)
                rows.append([baseline, spl, success_count, len(results[baseline]), np.mean(disc_returns), np.mean(time_estimates)])
            else:
                # one of the baselines does not have valid result (e.g. path to
                # target not found).  will skip this scene-target object setting
                # for all baselines so as to make sure we only use results that
                # are comparable between all baselines.
                return []

        cls.sharedheader = ["baseline", "spl", "success", "total", "disc_return", "time_est"]
        return rows

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        # Gathered results maps from global name to what is returned by gather()
        all_rows = []
        for global_name in gathered_results:
            scene_type, scene, target = global_name.split("-")
            for row in gathered_results[global_name]:
                success_rate = row[2] / max(1, row[3])  # success / total
                all_rows.append([scene_type, scene, target] + row + [success_rate])
        columns = ["scene_type", "scene", "target"] + cls.sharedheader + ["success_rate"]
        df_raw = pd.DataFrame(all_rows, columns=columns)

        ci_func = lambda x: ci_normal(x, confidence_interval=0.95)
        summary = df_raw.groupby(['baseline'])\
                        .agg([("avg", "mean"),
                              "std",
                              ("ci95", ci_func)])
        summary_by_scene_type = df_raw.groupby(['scene_type', 'baseline'])\
                                      .agg([("avg", "mean"),
                                            "std",
                                            ("ci95", ci_func),
                                            ("count", "sum")])
        summary_by_scene = df_raw.groupby(['scene', 'baseline'])\
                                 .agg([("avg", "mean"),
                                       "std",
                                       ("ci95", ci_func)])
        summary_by_target = df_raw.groupby(['target', 'baseline'])\
                                  .agg([("avg", "mean"),
                                        "std",
                                        ("ci95", ci_func),
                                        ("count", "sum")])
        df_raw.to_csv(os.path.join(path, "path_result.csv"))
        summary.to_csv(os.path.join(path, "path_result_summary.csv"))
        summary_by_scene_type.to_csv(os.path.join(path, "path_result_summary-by-scene-type.csv"))
        summary_by_scene.to_csv(os.path.join(path, "path_result_summary-by-scene.csv"))
        summary_by_target.to_csv(os.path.join(path, "path_result_summary-by-target.csv"))

        #######################################################################
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
            for scene_type in sorted(SCENE_TYPES):
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

        #######################################################################
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

        #######################################################################
        # Statistical significance
        ## SPL
        print("--------------------------------------------------")
        ### total (SPL)
        spl_totals = PathResult._filter_results_and_organize_by_method(df_raw, "spl")
        PathResult._print_statistical_significance_matrix(
            spl_totals, forwhat="total SPL", sigstr=SIGSTR)
        print("--------------------------------------------------")
        ### scene-type-wise (SPL)
        for scene_type in sorted(SCENE_TYPES):
            scene_type = scene_type.replace('_', '+')
            spl_scenes = PathResult._filter_results_and_organize_by_method(
                df_raw, "spl", filter_func=lambda row: row["scene_type"] == scene_type)
            PathResult._print_statistical_significance_matrix(
                spl_scenes, forwhat="{} SPL".format(scene_type), sigstr=SIGSTR)
        ### target-wise (SPL)
        for scene_type in sorted(SCENE_TYPES):
            for target in OBJECT_CLASSES[scene_type]['target']:
                scene_type = scene_type.replace('_', '+')
                spl_targets = PathResult._filter_results_and_organize_by_method(
                    df_raw, "spl", filter_func=lambda row: row["scene_type"] == scene_type and row["target"] == target)
                PathResult._print_statistical_significance_matrix(
                    spl_targets, forwhat="{}, {} SPL".format(scene_type, target), sigstr=SIGSTR)

        ## DR (discounted return)
        print("--------------------------------------------------")
        ### total (DR)
        dr_totals = PathResult._filter_results_and_organize_by_method(df_raw, "disc_return")
        PathResult._print_statistical_significance_matrix(
            dr_totals, forwhat="total DR", sigstr=SIGSTR)
        print("--------------------------------------------------")
        ### scene-type-wise (DR)
        for scene_type in sorted(SCENE_TYPES):
            scene_type = scene_type.replace('_', '+')
            dr_scenes = PathResult._filter_results_and_organize_by_method(
                df_raw, "disc_return", filter_func=lambda row: row["scene_type"] == scene_type)
            PathResult._print_statistical_significance_matrix(
                dr_scenes, forwhat="{} DR".format(scene_type), sigstr=SIGSTR)
        ### target-wise (DR)
        for scene_type in sorted(SCENE_TYPES):
            for target in OBJECT_CLASSES[scene_type]['target']:
                scene_type = scene_type.replace('_', '+')
                dr_targets = PathResult._filter_results_and_organize_by_method(
                    df_raw, "disc_return", filter_func=lambda row: row["scene_type"] == scene_type and row["target"] == target)
                PathResult._print_statistical_significance_matrix(
                    dr_targets, forwhat="{}, {} DR".format(scene_type, target), sigstr=SIGSTR)
        print("--------------------------------------------------")

        #######################################################################
        # Statical Significance specifically for hard-to-detect objects
        ## load the detector_params.csv
        import csv
        path_to_detector_params_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "../../experiments/thor/detector_params.csv")
        hard_to_detect_targets = set()
        print("Hard to Detect Classes:")
        with open(path_to_detector_params_csv) as f:
            reader = csv.DictReader(f)
            for csvrow in reader:
                if csvrow['class'] in OBJECT_CLASSES[csvrow['scene_type']]["target"]:
                    tf_rate = float(csvrow['TP_rate'])
                    if tf_rate < HARD_TO_DETECT_TP:
                        print("- {}: {:.3f}".format(csvrow['class'], tf_rate))
                        hard_to_detect_targets.add((csvrow['scene_type'], csvrow['class']))

        # note that df_raw is the dataframe that contains ALL individual results without any aggregation..
        spl_hard_to_detect = PathResult._filter_results_and_organize_by_method(
            df_raw, "spl", filter_func=lambda row: (row["scene_type"], row["target"]) in hard_to_detect_targets)
        PathResult._print_statistical_significance_matrix(
            spl_hard_to_detect, forwhat="Hard to Detect SPL".format(scene_type, target), sigstr=SIGSTR)

        dr_hard_to_detect = PathResult._filter_results_and_organize_by_method(
            df_raw, "disc_return", filter_func=lambda row: (row["scene_type"], row["target"]) in hard_to_detect_targets)
        PathResult._print_statistical_significance_matrix(
            dr_hard_to_detect, forwhat="Hard to Detect DR".format(scene_type, target), sigstr=SIGSTR)

        time_hard_to_detect = PathResult._filter_results_and_organize_by_method(
            df_raw, "time_est", filter_func=lambda row: (row["scene_type"], row["target"]) in hard_to_detect_targets)
        PathResult._print_statistical_significance_matrix(
            dr_hard_to_detect, forwhat="Time Estimate".format(scene_type, target), sigstr=SIGSTR)
        _df_time = pd.DataFrame({"method": list(time_hard_to_detect.keys()),
                                 "avg_time_est": [np.nanmean(time_hard_to_detect[method])
                                                  for method in time_hard_to_detect]})
        print(_df_time)

    @staticmethod
    def _filter_results_and_organize_by_method(df_complete, metric, filter_func=None):
        """Given a dataframe that contains all result rows, returns a dictionary that
        maps from method name (e.g. 'Random') to a series that contains only
        the result of given metric (e.g. 'spl'), such that this series is filtered
        is according to a criteria specified by the given filtering function.

        filter_func takes in row (pandas row) and returns True or False.

        Note that the method name is the output of baseline_name() function.
        """
        methods = df_complete["baseline"].unique()
        result = {}  # maps from method name to series.
        if filter_func is not None:
            m = df_complete.apply(filter_func, axis=1)
            df_filtered = df_complete[m]
        else:
            df_filtered = df_complete
        for baseline in methods:
            df_baseline = df_filtered.loc[df_filtered['baseline'] == baseline]
            # Note that df_baseline[metric] will preserve the order
            # of the result values. This is important for non-parametric
            # tests
            result[baseline_name(baseline)] = df_baseline[metric]
        return result

    @staticmethod
    def _print_statistical_significance_matrix(results_by_method, forwhat="", sigstr=True):
        """
        utility function to print statistical significance matrix.
        Args:
            results_by_method: dictionary mapping from baseline name (e.g. Random)
                to a list of result values (for example, all the SPL values for all trials).
            forwhat (str): some info to print before printing the matrix
            sigstr (bool): If true, in the matrix will show 'ns', '*', '**', etc. and
                otherwise show the p value.
            sigmethod (str): Method of statistical testing. Either 't_ind' or 'wilcoxon'
        """
        print("\nStatistical significance ({}):".format(forwhat))
        dfsig = test_significance_pairwise(results_by_method, sigstr=sigstr, method=SIGMETHOD)
        print("** For {}**".format(forwhat))
        print(dfsig)
        print("\n")


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
