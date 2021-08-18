import os
import copy
import random
import math

import thortils

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial

from datetime import datetime as dt
from sciex import Experiment

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ABS_PATH, "../../", "results")

POUCT_ARGS = dict(max_depth=30,
                  num_sims=500,
                  discount_factor=0.95,
                  exploration_const=100)

def make_trial(run_num, scene, target, other, detectors, corr=None):
    """If corr is None, then will make use of correlation;
    Otherwise, will only use target detector (i.e. ignore detection
    of other objects)"""
    if corr is None:
        detectables = {target}
    else:
        detectables = {target, other}

    args = TaskArgs(detectables=detectables,
                    scene=scene,
                    target=target,
                    max_steps=200,
                    agent_class="ThorObjectSearchCosAgent",
                    task_env="ThorObjectSearch")
    config = make_config(args)
    config["agent_config"]["corr_specs"] = {}
    if corr is not None:
        config["agent_config"]["corr_specs"] = {
            (target, other): corr
        }

    config["agent_config"]["detector_specs"] = {
        cls: detectors[cls]
        for cls in args.detectables
    }
    config["agent_config"]["solver"] = "pomdp_py.POUCT"
    config["agent_config"]["solver_args"] = POUCT_ARGS
    config["visualize"] = False
    if corr is None:
        trial_name = f"{scene}-{target}-{other}_{run_num:0>3}_target-only"
    else:
        trial_name = f"{scene}-{target}-{other}_{run_num:0>3}_corr"
    trial = ThorObjectSearchTrial(trial_name, config, verbose=True)
    return trial

from FloorPlanSettings import SETTINGS
def EXPERIMENT_tiny(split=3, num_trials=5):
    scenes = ["FloorPlan1", "FloorPlan2", "FloorPlan3"]

    all_trials = []
    for scene in scenes:
        settings = SETTINGS[scene]
        for combo in settings['combos']:
            target, other, corr = combo
            detectors = settings['detectors']
            print(f"Creating trials for {target}--{other}")
            for i in range(num_trials):
                target_only_trial = make_trial(i+1, scene, target, other, detectors, corr=None)
                corr_trial = make_trial(i+1, scene, target, other, detectors, corr=corr)
                all_trials.append(target_only_trial)
                all_trials.append(corr_trial)

    exp_name = "ExperimentTiny-BB"
    exp = Experiment(exp_name,
                     all_trials,
                     OUTPUT_DIR,
                     verbose=True,
                     add_timestamp=True)
    exp.generate_trial_scripts(split=split)
    print("Trials generated at %s/%s" % (exp._outdir, exp.name))
    print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    EXPERIMENT_tiny(split=4, num_trials=5)
