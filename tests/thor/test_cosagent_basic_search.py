import thortils

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial


def _test_basic_search():
    args = TaskArgs(detectables={"Fridge", "Bread"},
                    scene='FloorPlan1',
                    target="Fridge",
                    agent_class="ThorObjectSearchCosAgent",
                    task_env="ThorObjectSearch")
    config = make_config(args)
    config["agent_config"]["corr_specs"] = {
        ("Fridge", "Bread"): (around, dict(d=1))
    }
    config["agent_config"]["detector_specs"] = {
        "Fridge": ("fan-nofp", dict(fov=45, min_range=1, max_range=3), (0.7, 0.1)),
        "Bread": ("fan-nofp", dict(fov=90, min_range=1, max_range=4), (0.7, 0.1))
    }
    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_cosagent", config, verbose=True)
    print("Trial created")
    trial.run()


if __name__ == "__main__":
    _test_basic_search()
