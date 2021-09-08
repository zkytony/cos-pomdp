# One of the most critical tests. Loads a vision detector,
# and uses it for search. Also, uses the kind of correlation
# information specified in the experiment setup. See how this works.

import os
import sys

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ABS_PATH, "../../experiments/thor"))

import thortils as tt

from experiment_thor import (Methods,
                             make_trial,
                             OBJECT_CLASSES,
                             read_detector_params)

def prepare(scene_type):
    detector_models = read_detector_params()
    targets = OBJECT_CLASSES[scene_type]["target"]
    corr_objects = OBJECT_CLASSES[scene_type]["corr"]
    return detector_models, targets, corr_objects


def _test_method(method, scene_type, target_class, scene="FloorPlan21"):
    valscenes = tt.ithor_scene_names(scene_type, levels=range(21, 31))
    if scene not in valscenes:
        raise ValueError("Only allow validation scenes.")

    detector_models, targets, corr_objects = prepare(scene_type)
    if target_class not in targets:
        raise ValueError("{} is not a valid target class".format(target_class))

    trial = make_trial(method, 0, scene_type, scene, target_class,
                       detector_models, corr_objects=corr_objects,
                       visualize=True)
    trial.run()


if __name__ == "__main__":
    _test_method(Methods.HIERARCHICAL_CORR_CRT, "kitchen", "SaltShaker")
