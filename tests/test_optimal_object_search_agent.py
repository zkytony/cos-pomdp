# Note:
from cosp.thor.trial import build_object_search_trial
from cosp.thor.utils import compute_spl
from pprint import pprint

TARGETS = {
    "FloorPlan1": ["Vase", "Bread", "Book", "Lettuce"],
    "FloorPlan2": ["Bowl", "Pan", "Ladle"],
    "FloorPlan3": ["Bread", "SoapBottle", "Spatula"],
    "FloorPlan4": ["SaltShaker", "SinkBasin", "Pan"],
    "FloorPlan5": ["Pot", "CoffeeMachine", "Faucet"],
}

def test_many(targets):
    all_results = []
    for floorplan in targets:
        for target in targets[floorplan]:
            all_results.append(collect(test_singe(floorplan, target)))
            print(floorplan, target, all_results[-1].to_tuple())
    spl, sr = gather(all_results)
    print("SPL: {:.3f}".format(spl))
    print("SR: {:.4f}".format(sr))

def test_singe(floorplan, object_type):
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
    return spl, success_rate

if __name__ == "__main__":
    # test_out_optimal_agent()
    test_many(TARGETS)
