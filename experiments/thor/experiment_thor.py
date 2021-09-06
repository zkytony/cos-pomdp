import os
import copy
import random
import math

import thortils

from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor import agent as agentlib
from cospomdp_apps.thor.trial import ThorObjectSearchTrial

from datetime import datetime as dt
from sciex import Experiment

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

class Methods:
    HIERARCHICAL_CORR_GT = dict("ThorObjectSearchCompleteCosAgent", use_corr=True, corr_type="groundtruth")
    HIERARCHICAL_CORR_LRN = dict("ThorObjectSearchCompleteCosAgent", use_corr=True, corr_type="learned")
    HIERARCHICAL_CORR_WRG = dict("ThorObjectSearchCompleteCosAgent", use_corr=True, corr_type="wrong")
    HIERARCHICAL_TARGET = dict("ThorObjectSearchCompleteCosAgent", use_corr=False)
    FLAT_POUCT_CORR_GT = dict("ThorObjectSearchBasicCosAgent", use_corr=False, corr_type="groundtruth")
    GREEDY_NBV = dict("ThorObjectSearchGreedyNbvAgent", use_corr=True, corr_type="groundtruth")
    RANDOM = dict("ThorObjectSearchRandomAgent", use_corr=False)

def make_trial(scene, target, corr_objects, correlations, detector_models, method):
    """
    Args:
        scene: scene to search in
        target: object class to search for
        corr_objects (list): objects used as correlated objects to help
        correlations: (some kind of data structure that conveys the correlation information),
        detector_models (dict): Maps from object class to a detector models used for POMDP planning
        method: (method to use to search)
    """
    detectables = set(target) | set(corr_objects)

    args = TaskArgs(detectables=detectables,
                    scene=scene,
                    target=target,
                    agent_class=)
