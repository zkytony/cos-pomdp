# Note:
import random
from cosp.thor.trial import build_object_search_trial
from cosp.thor.utils import compute_spl
from pprint import pprint

####### KITCHEN ##########
TARGETS_EXPOSED = {
    "FloorPlan1": ["Vase", "Bread", "Book", "Lettuce"],
    "FloorPlan2": ["Fork", "Pan", "Ladle"],
    "FloorPlan3": ["Bread", "SoapBottle", "Spatula"],
    "FloorPlan4": ["SaltShaker", "SinkBasin", "Pan"],
    "FloorPlan5": ["Knife", "CoffeeMachine", "Faucet"],
}

TARGETS_CONTAINED = {
    "FloorPlan1": ["Knife", "Egg", "Cup"],
    "FloorPlan2": ["Plate", "Apple", "Butterknife"],
    "FloorPlan3": ["Pan", "Tomato"],
    "FloorPlan4": ["Egg", "Lettuce"],
    "FloorPlan5": ["Apple"]
}

####### BATHROOM ##########
TARGETS_EXPOSED = {
    "FloorPlan402": ["Candle", "Plunger", "SoapBar"],
    "FloorPlan403": ["Glass", "Faucet", "Towel"],
    "FloorPlan405": ["LightSwitch", "ScrubBrush", "HandTowelHolder"],
    "FloorPlan407": ["Cloth", "SinkBasin", "Toilet"],
    "FloorPlan409": ["ShowerHead", "TowelHolder", "Mirror"],
}

TARGETS_CONTAINED = {
    "FloorPlan402": ["TissueBox", "ToiletPaper"],
    "FloorPlan403": ["PaperTowelRoll", "DishSponge"],
    "FloorPlan405": ["Candle"],
    "FloorPlan407": ["SprayBottle"],
    "FloorPlan409": ["ToiletPaper"]
}

def test_many(targets):
    all_results = []
    for floorplan in targets:
        for target in targets[floorplan]:
            all_results.append(collect(test_singe(floorplan, target)))
            print(floorplan, target, all_results[-1].to_tuple())
    spl, sr, failed_objects = gather(all_results)
    print("SPL: {:.3f}".format(spl))
    print("SR: {:.4f}".format(sr))
    print("Failed objects:")
    pprint(failed_objects)

def test_singe(floorplan, object_type):
    print("Searching for {} in {}".format(object_type, floorplan))
    trial = build_object_search_trial(floorplan, object_type, "class")
    return trial.run(logging=True)

def collect(trial_result):
    return trial_result[0]  # pathresult

def gather(all_results):
    episode_results = [r.to_tuple()
                       for r in all_results]
    spl = compute_spl(episode_results)

    success_count = sum(1 for r in all_results
                        if r.success is True)
    success_rate = success_count / len(all_results)
    failed_objects = [(r.scene, r.target) for r in all_results
                      if r.success is False]
    return spl, success_rate, failed_objects

if __name__ == "__main__":
    # test_out_optimal_agent()
    test_many(TARGETS)
