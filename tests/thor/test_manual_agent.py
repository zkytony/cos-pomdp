import sys
import os
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor.common import TaskArgs, make_config
import thortils as tt

# Use experiment setup
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ABS_PATH, "../../experiments/thor"))
from experiment_thor import OBJECT_CLASSES

def _test_manual_agent(target,
                       scene="FloorPlan1",
                       max_steps=100):
    print("Test cospomdp_keyboard manual agent")
    scene_type = tt.ithor_scene_type(scene)
    detectables = OBJECT_CLASSES[scene_type]["corr"] + [target]

    agent_init_inputs = ["grid_map"]
    args = TaskArgs(detectables=detectables,
                    scene=scene,
                    target=target,
                    agent_class="ThorObjectSearchKeyboardAgent",
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    agent_init_inputs=agent_init_inputs,
                    use_vision_detector=True,
                    plot_detections=True)
    config = make_config(args)

    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_manual", config, verbose=True)
    print("Trial created")
    trial.run()

if __name__ == "__main__":
    _test_manual_agent('PepperShaker',
                       scene="FloorPlan1")
