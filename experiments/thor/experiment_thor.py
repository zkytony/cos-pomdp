import os
import copy
import random
import math
import json
import pandas as pd
from datetime import datetime as dt

from sciex import Experiment
import thortils as tt

from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor import agent as agentlib
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor import constants
from cospomdp.utils.corr_funcs import ConditionalSpatialCorrelation

# Configurations
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ABS_PATH, "../../", "results")

POUCT_ARGS = dict(max_depth=30,
                  num_sims=200,
                  discount_factor=0.95,
                  exploration_const=100,
                  show_progress=True)

LOCAL_POUCT_ARGS = POUCT_ARGS
MAX_STEPS = 100

TOPO_PLACE_SAMPLES = 20  # specific to hierarchical methods

class Methods:
    # correct: the distance used to form spatial correlation comes from the actual distance between instances
    # in the validation scene directly, instead of learned from training as _LRN does.
    HIERARCHICAL_CORR_CRT = dict(agent="ThorObjectSearchCompleteCosAgent", use_corr=True, corr_type="correct")
    HIERARCHICAL_CORR_LRN = dict(agent="ThorObjectSearchCompleteCosAgent", use_corr=True, corr_type="learned")
    HIERARCHICAL_CORR_WRG = dict(agent="ThorObjectSearchCompleteCosAgent", use_corr=True, corr_type="wrong")
    HIERARCHICAL_TARGET = dict(agent="ThorObjectSearchCompleteCosAgent", use_corr=False)
    FLAT_POUCT_CORR_CRT = dict(agent="ThorObjectSearchBasicCosAgent", use_corr=True, corr_type="correct")
    FLAT_POUCT_TARGET_CRT = dict(agent="ThorObjectSearchBasicCosAgent", use_corr=False, corr_type="correct")
    GREEDY_NBV = dict(agent="ThorObjectSearchGreedyNbvAgent", use_corr=True, corr_type="correct")
    RANDOM = dict(agent="ThorObjectSearchRandomAgent", use_corr=False)

    @staticmethod
    def get_name(method):
        if "random" in method['agent'].lower():
            return "random"
        if "greedy" in method['agent'].lower():
            return "greedy-nbv"
        if "basic" in method['agent'].lower():
            if not method['use_corr']:
                return "flat#target-only"
            else:
                assert method['corr_type'] == "correct"
                return "flat#corr"
        if "complete" in method['agent'].lower():
            if not method["use_corr"]:
                return "hierarchical#target-only"
            else:
                if method["corr_type"] == "correct":
                    return "hierarchical#corr"
                elif method["corr_type"] == "learned":
                    return "hierarchical#corr-learned"
                elif method["corr_type"] == "wrong":
                    return "hierarchical#corr-wrong"
                else:
                    raise ValueError("Does not understand correlation type: {}".format(method["corr_type"]))

OBJECT_CLASSES = {
    "kitchen": {"target": ["SaltShaker", "Mug", "DishSponge"],
                "corr": ["StoveBurner", "Microwave", "CoffeeMachine", "Fridge", "SoapBottle"]},
    "living_room": {"target": ["KeyChain", "CreditCard", "Laptop"],
                    "corr": ["FloorLamp", "HousePlant", "Television", "Painting", "Sofa"]},
    "bedroom": {"target": ["CellPhone", "Book", "CD"],
                "corr": ["DeskLamp", "Laptop", "Mirror", "Pillow", "GarbageCan"]},
    "bathroom": {"target": ["Candle", "ScrubBrush", "Plunger"],
                 "corr": ["Toilet", "Towel", "Mirror", "HandTowel", "SprayBottle"]}
}

def make_trial(method, run_num, scene_type, scene,
               target, detector_models,
               corr_objects=None,
               max_steps=constants.MAX_STEPS,
               rnd=random, visualize=False, viz_res=20):
    """
    Args:
        scene: scene to search in
        target: object class to search for
        corr_objects (list): objects used as correlated objects to help
        correlations: (some kind of data structure that conveys the correlation information),
        detector_models (dict): Maps from object class to a detector models configuration used for POMDP planning;
            e.g. {"Apple": dict(fov=90, min_range=1, max_range=target_range), (target_accuracy, 0.1))}

        method_name: a string, e.g. "HIERARCHICAL_CORR_CRT"
    """
    if corr_objects is None:
        corr_objects = set()
    detectables = set({target}) | set(corr_objects)

    agent_init_inputs = []
    if method["agent"] != "ThorObjectSearchRandomAgent":
        agent_init_inputs = ['grid_map', 'agent_pose']

    args = TaskArgs(detectables=detectables,
                    scene=scene,
                    target=target,
                    agent_class=method["agent"],
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    agent_init_inputs=agent_init_inputs)
    config = make_config(args)
    config["agent_config"]["corr_specs"] = {}
    config["agent_config"]["detector_specs"] = {
        target: detector_models[target]
    }

    if method["use_corr"]:
        for other in corr_objects:
            spcorr = load_correlation(scene, scene_type, target, other, method["corr_type"], rnd=rnd)
            config["agent_config"]["corr_specs"][(target, other)] = (spcorr.func, {}) # corr_func, corr_func_args
            config["agent_config"]["detector_specs"][other] = detector_models[other]

    config["agent_config"]["solver"] = "pomdp_py.POUCT"
    config["agent_config"]["solver_args"] = POUCT_ARGS

    if "CompleteCosAgent" in method['agent']:
        config["agent_config"]["num_place_samples"] = TOPO_PLACE_SAMPLES
        config["agent_config"]["local_search_type"] = "basic"
        config["agent_config"]["local_search_params"] = LOCAL_POUCT_ARGS

    config["visualize"] = visualize
    config["viz_config"] = {
        'res': viz_res
    }
    trial_name = f"{scene_type}-{scene}-{target}_{run_num:0>3}_{Methods.get_name(method)}"
    trial = ThorObjectSearchTrial(trial_name, config, verbose=True)
    return trial

def read_detector_params(filepath=os.path.join(ABS_PATH, "detector_params.csv")):
    detector_models = {}
    df = pd.read_csv(filepath)
    for scene_type in OBJECT_CLASSES:
        for cls in (OBJECT_CLASSES[scene_type]['target'] + OBJECT_CLASSES[scene_type]['corr']):
            row = df.loc[(df['scene_type'] == scene_type) & (df['class'] == cls)]
            quality_params = (row["TP_rate"], row["FP_rate"], 0.1)
            max_range = row["dist"] / constants.GRID_SIZE
            detector_models[cls] = ("fan-simplefp",
                                    dict(fov=constants.FOV, min_range=1, max_range=max_range),
                                    quality_params)
    return detector_models

CORR_DATASET = os.path.join(ABS_PATH, "../../data/thor/corrs")
def load_correlation(scene, scene_type, target, corr_object, corr_type, rnd=random):
    if corr_type == "correct":
        fname = f"distances_{scene_type}_{target}-{corr_object}_{scene}.json"
    elif corr_type == "learned":
        fname = f"distances_{scene_type}_{target}-{corr_object}_train.json"
    elif corr_type == "wrong":
        # For the wrong correlation, instead of taking target and corr_object,
        # we randomly choose another corr_object instead, as the correlation
        # for the given `target`, `corr_object`. This will be a wrong correlation,
        # but not the worst.
        another_corr_object = rnd.sample(OBJECT_CLASSES[scene]['corr'], 1)[0]
        fname = f"distances_{scene_type}_{target}-{another_corr_object}_{scene}.json"
    else:
        raise ValueError("Unknown corr type {}".format(corr_type))

    with open(os.path.join(CORR_DATASET, fname)) as f:
        dd = json.load(f)
    spcorr = ConditionalSpatialCorrelation(target, corr_object, dd["distances"])
    return spcorr


def EXPERIMENT_THOR(split=10, num_trials=3):
    """
    Each object is search `num_trials` times
    """
    all_trials = []
    rnd = random.Random(1000) # this is used to generated deterministic trials
    for scene_type in ['kitchen', 'living_room', 'bedroom', 'bathroom']:
        for scene in tt.ithor_scene_names(scene_type, levels=(21,31)):  # use the last 10 for evaluation

            targets = OBJECT_CLASSES[scene]["target"]
            corr_objects = OBJECT_CLASSES[scene]["corr"]

            # make detector models
            detector_models = read_detector_params()

            for target, true_positive_rate, avg_detection_range in targets:

                for run_num in range(num_trials):
                    hier_corr_crt = make_trial(Methods.HIERARCHICAL_CORR_CRT,
                                               run_num, scene_type, scene,
                                               target, detector_models,
                                               corr_objects=corr_objects,
                                               rnd=rnd)

                    hier_corr_lrn = make_trial(run_num, scene_type, scene, target, corr_objects,
                                              detector_models, Methods.HIERARCHICAL_CORR_LRN)
                    hier_corr_wrg = make_trial(run_num, scene_type, scene, target, corr_objects,
                                              detector_models, Methods.HIERARCHICAL_CORR_WRG)
                    hier_target = make_trial(run_num, scene_type, scene, target, corr_objects,
                                             detector_models, Methods.HIERARCHICAL_TARGET)
                    flat_corr_crt = make_trial(run_num, scene_type, scene, target, corr_objects,
                                              detector_models, Methods.FLAT_POUCT_CORR_CRT)
                    flat_target_crt = make_trial(run_num, scene_type, scene, target, corr_objects,
                                                detector_models, Methods.FLAT_POUCT_TARGET_CRT)
                    greedynbv = make_trial(run_num, scene_type, scene, target, corr_objects,
                                           detector_models, Methods.GREEDY_NBV)
                    random = make_trial(run_num, scene_type, scene, target, corr_objects,
                                        detector_models, Methods.RANDOM)
                    all_trials.extend([hier_corr_crt,
                                       hier_target,
                                       flat_corr_crt,
                                       flat_target_crt,
                                       greedynbv,
                                       random])
    exp_name = "ExperimentThor-AA"
    exp = Experiment(exp_name,
                     all_trials,
                     OUTPUT_DIR,
                     verbose=True,
                     add_timestamp=True)
    exp.generate_trial_scripts(split=split, use_mp=True)
    print("Trials generated at %s/%s" % (exp._outdir, exp.name))
    print("Find multiple computers to run these experiments.")

if __name__ == "__main__":
    EXPERIMENT_THOR(split=10, num_trials=3)
