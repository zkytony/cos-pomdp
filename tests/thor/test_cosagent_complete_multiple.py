from pomdp_py.utils import TreeDebugger
from test_cosagent_complete import _test_complete_search

# Test multiple instances. Observe what is working and what is not

planning_params = dict(
    max_depth=30,
    num_sims=200,
    max_steps=50,
    discount_factor=0.95,
    exploration_const=100,
    num_place_samples=20
)

task_setup = {
    # Kitchens
    # "FloorPlan1-1": dict(target="Spatula",
    #                      other="StoveBurner",
    #                      dist=3,
    #                      target_range=5,
    #                      other_range=6,
    #                      target_accuracy=0.7,
    #                      other_accuracy=0.8),
    # "FloorPlan1-2": dict(target="Bread",
    #                      other="CounterTop",
    #                      dist=3,
    #                      target_range=4,
    #                      other_range=6,
    #                      target_accuracy=0.5,
    #                      other_accuracy=0.8),
    # "FloorPlan2-1": dict(target="Mug",
    #                      other="CounterTop",
    #                      dist=4,
    #                      target_range=4,
    #                      other_range=6,
    #                      target_accuracy=0.6,
    #                      other_accuracy=0.8),
    # "FloorPlan3-1": dict(target="CoffeeMachine",
    #                      other="Microwave",
    #                      dist=4,
    #                      target_range=5,
    #                      other_range=6,
    #                      target_accuracy=0.6,
    #                      other_accuracy=0.8),
    "FloorPlan4-1": dict(target="Spatula",
                         other="Microwave",
                         dist=3,
                         target_range=4,
                         other_range=6,
                         target_accuracy=0.4,
                         other_accuracy=0.8),
    # Living Room
}


def _test_cosagent_complete_multiple():
    for scenestr in task_setup:
        scene = scenestr.split("-")[0]
        setup = task_setup[scenestr]

        _test_complete_search(scene=scene,
                              **setup,
                              **planning_params)

if __name__ == "__main__":
    _test_cosagent_complete_multiple()
