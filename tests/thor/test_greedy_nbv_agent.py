import os
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp.utils.corr_funcs import around

def _test_greedy_agent(target,
                       other=None,
                       scene="FloorPlan1",
                       dist=3,
                       target_range=5,
                       other_range=8,
                       target_accuracy=0.7,
                       other_accuracy=0.8,
                       target_false_pos=0.15,
                       other_false_pos=0.1,
                       use_vision_detector=False,
                       save=False,
                       max_steps=100):
    print("Test cospomdp_random agent")
    agent_init_inputs = ["grid_map", "agent_pose"]
    if other is None:
        detectables = [target]
    else:
        detectables = [target, other]

    detector_specs = {
        target: ("fan-far", dict(fov=90, min_range=1, mean_range=target_range),
                 (target_accuracy, target_false_pos, 0.1))
    }
    corr_specs = {}
    if other is not None:
        corr_specs[(target, other)] = (around, dict(d=dist))
        detector_specs[other] =\
            ("fan-far", dict(fov=90, min_range=1, mean_range=other_range),
             (other_accuracy, other_false_pos, 0.1))

    args = TaskArgs(detectables=detectables,
                    scene='FloorPlan1',
                    target=target,
                    agent_class="ThorObjectSearchGreedyNbvAgent",
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    agent_init_inputs=agent_init_inputs,
                    use_vision_detector=use_vision_detector,
                    plot_detections=True,
                    agent_detector_specs=detector_specs,
                    corr_specs=corr_specs)
    config = make_config(args)

    config["agent_config"]["num_particles"] = 200

    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    if save:
        method = "#corr" if other is not None else "#target-only"
        config['task_config']['detector_config']['plot_detections'] = True
        config['save_path'] = os.path.join(f"./test-{scene}-{target}-greedy-nbv{method}", "vis")
        config['save_opts'] = {'gif': True,
                               'duration': 0.25}
    trial = ThorObjectSearchTrial("test_greedy", config, verbose=True)
    print("Trial created")
    trial.run()

if __name__ == "__main__":
    _test_greedy_agent('PepperShaker', 'StoveBurner',
                       use_vision_detector=True,
                       save=True)
