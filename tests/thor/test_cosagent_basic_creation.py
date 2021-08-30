import thortils

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchBasicCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial

def _test_create_trial():
    args = TaskArgs(detectables={"Apple", "CounterTop", "Bread"},
                    scene='FloorPlan1',
                    target="Apple",
                    agent_class="ThorObjectSearchBasicCosAgent",
                    task_env="ThorObjectSearch",
                    agent_init_inputs=['grid_size', 'grid_map', 'agent_pose'])
    config = make_config(args)
    config["agent_config"]["corr_specs"] = {
        ("Apple", "CounterTop"): (around, dict(d=3)),
        ("Apple", "Bread"): (around, dict(d=1))
    }
    config["agent_config"]["detector_specs"] = {
        "Apple": ("fan-nofp", dict(fov=45, min_range=1, max_range=3), (0.7, 0.1)),
        "CounterTop": ("fan-nofp", dict(fov=90, min_range=1, max_range=5), (0.8, 0.1)),
        "Bread": ("fan-nofp", dict(fov=90, min_range=1, max_range=4), (0.7, 0.1))
    }
    trial = ThorObjectSearchTrial("test_cosagent", config)
    print("Trial created")

def _test_cosagent_basic_creation():
    print("Test cosagent creation")
    _test_create_trial()

if __name__ == "__main__":
    _test_cosagent_basic_creation()
