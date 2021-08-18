# Note:
import random
from pprint import pprint
import thortils
from thortils import compute_spl
from cospomdp_apps.thor.trial import build_object_search_trial
from cospomdp.utils.math import mean_ci_normal


# ####### KITCHEN ##########
# # Notes:
# # Fork in FloorPlan2 doesn't work because it'll be blocked by wall
TARGETS_EXPOSED = {
    # "FloorPlan1": ["Vase", "Book", "Lettuce"],  # Book pitch wrong (due to height)
    # "FloorPlan2": ["Bowl", "Pan", "Ladle"],      # Ladle pitch wrong (due to height)
    "FloorPlan3": ["DishSponge"],#["Bread", "SoapBottle", "Spatula"],
    # "FloorPlan4": ["SaltShaker", "SinkBasin", "Pan"],
    # "FloorPlan5": ["Knife", "CoffeeMachine", "Faucet"],
}

# TARGETS_CONTAINED = {
#     "FloorPlan1": ["Knife", "Egg", "WineBottle"],  # All works
#     "FloorPlan2": ["Plate", "Apple", "ButterKnife"],  # Plate works, Apple: Blocked by door.
#     "FloorPlan3": ["Pan", "Tomato"],   # Pan: view blocked by table; Tomato: Fridge open blocked
#     "FloorPlan4": ["Egg", "Lettuce"],  # Fridge and Lettuce: fridge open blocked
#     "FloorPlan5": ["Apple"]  # Fridge open blocked
# }

# ####### BATHROOM ##########
# TARGETS_EXPOSED = {
#     "FloorPlan402": ["Candle", "Plunger"],#, "SoapBar"],  # SoapBar fails
#     "FloorPlan403": ["Glass", "Faucet", "Towel"],
#     "FloorPlan405": ["LightSwitch", "ScrubBrush", "HandTowelHolder"],
#     "FloorPlan407": ["Cloth", "SinkBasin", "Toilet"],
#     "FloorPlan409": ["ShowerHead", "TowelHolder", "Mirror"],
# }

# TARGETS_CONTAINED = {
#     "FloorPlan402": ["TissueBox", "ToiletPaper"],
#     "FloorPlan403": ["PaperTowelRoll", "DishSponge"],
#     "FloorPlan405": ["Candle"],
#     "FloorPlan407": ["SprayBottle"],
#     "FloorPlan409": ["ToiletPaper"]
# }

def _test_many(targets):
    all_results = []
    for floorplan in targets:
        for target in targets[floorplan]:
            all_results.append(collect(_test_singe(floorplan, target)))
            print(floorplan, target, all_results[-1]["path"].to_tuple())
    spl, sr, failed_objects, disc_return = gather(all_results)
    print("********* RESULTS ***********")
    print("SPL: {:.3f}".format(spl))
    print("SR: {:.4f}".format(sr))
    print("Discounted Return: {} ({})".format(disc_return[0], disc_return[1]))
    print("Failed objects:")
    pprint(failed_objects)


def _test_singe_by_id(floorplan, object_id):
    print("** Searching for {} in {}".format(object_id, floorplan))
    trial = build_object_search_trial(floorplan, object_id, "object")
    return trial.run(logging=True)

def _test_singe(floorplan, object_type):
    print("** Searching for {} in {}".format(object_type, floorplan))
    trial = build_object_search_trial(floorplan, object_type, "class")
    return trial.run(logging=True)

def collect(trial_result):
    path_result = trial_result[0]  # pathresult
    history_result = trial_result[1]
    return dict(path=path_result, history=history_result)

def gather(all_results):
    episode_results = [r["path"].to_tuple() for r in all_results]
    spl = compute_spl(episode_results)
    success_count = sum(1 for r in all_results
                        if r["path"].success is True)
    success_rate = success_count / len(all_results)
    failed_objects = [(r["path"].scene, r["path"].target) for r in all_results
                      if r["path"].success is False]
    discounted_returns = [r["history"].discounted_return() for r in all_results]
    mean, ci = mean_ci_normal(discounted_returns)
    return spl, success_rate, failed_objects, (mean, ci)

def _debug_pitch():
    _test_singe_by_id("FloorPlan1", "Cabinet|+00.68|+02.02|-02.46")
    _test_singe_by_id("FloorPlan1", "Cabinet|+00.68|+00.50|-02.20")


if __name__ == "__main__":
    # _test_out_optimal_agent()
    _test_many(TARGETS_EXPOSED)
