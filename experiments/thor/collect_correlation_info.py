# Collect "distances", which will be used to build
# correlation models in experiments.
import os
import json
import thortils as tt
from tqdm import tqdm
from experiment_thor import OBJECT_CLASSES
from cospomdp_apps.thor import constants

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ABS_PATH, "../../", "data")





def collect_for(scene_type, for_train=True):
    # For each scene in training scenes,
    #   launch a controller.
    #   For each target-corr_object tuple
    #      count distances in that scene
    #      and save the distances
    if for_train:
        scenes = tt.ithor_scene_names(scene_type, levels=range(1, 21))
    else:
        scenes = tt.ithor_scene_names(scene_type, levels=range(21, 31))

    cc = {}  # maps {(target, corr_object) -> {scene -> [distances]}}
    for scene in tqdm(scenes):
        controller = tt.launch_controller({**constants.CONFIG, **{"scene": scene}})
        for target in OBJECT_CLASSES[scene_type]['target']:
            for corr_object in OBJECT_CLASSES[scene_type]['corr']:
                distances = tt.thor_distances_in_scene(controller.last_event, target, corr_object)
                key = (target, corr_object)
                if key not in cc:
                    cc[key] = {}
                cc[key][scene] = distances
        controller.stop()

    savedir = os.path.join(OUTPUT_DIR, "thor", "corrs")
    os.makedirs(savedir, exist_ok=True)
    if for_train:
        # combine all the distances and save a single file
        result = []
        for key in cc:
            all_distances = []
            for scene in cc[key]:
                all_distances.extend(cc[key][scene])
            target, corr_object = key
            result.append({"key": [target, corr_object],
                           "distances": all_distances})
            with open(os.path.join(savedir, f"distances_{scene_type}_{target}-{corr_object}_train.json"), 'w') as f:
                json.dump(result, f, indent=4, sort_keys=True)

    else:
        # save individually per validation scene
        for key in cc:
            target, corr_object = key
            for scene in cc[key]:
                distances = cc[key][scene]
                with open(os.path.join(savedir, f"distances_{scene_type}_{target}-{corr_object}_{scene}.json"), 'w') as f:
                    json.dump({"key": key,
                               "distances": distances}, f,
                              indent=4, sort_keys=True)

if __name__ == "__main__":
    collect_for("kitchen", for_train=True)
    collect_for("kitchen", for_train=False)
    collect_for("living_room", for_train=True)
    collect_for("living_room", for_train=False)
    collect_for("bedroom", for_train=True)
    collect_for("bedroom", for_train=False)
    collect_for("bathroom", for_train=True)
    collect_for("bathroom", for_train=False)
