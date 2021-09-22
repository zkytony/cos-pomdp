import thortils
import os

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.trial import ThorObjectSearchTrial

__all__ = ['_test_basic_search']

def _test_basic_search(target,
                       other,
                       prior='uniform',
                       scene="FloorPlan1",
                       dist=3,
                       target_range=7,
                       other_range=9,
                       target_accuracy=0.7,
                       other_accuracy=0.8,
                       target_false_pos=0.15,
                       other_false_pos=0.1,
                       max_depth=20,
                       num_sims=200,
                       max_steps=100,
                       discount_factor=0.95,
                       exploration_const=100,
                       use_vision_detector=True,
                       show_progress=True,
                       step_act_cb=None,
                       step_act_args={},
                       step_update_cb=None,
                       save=False):
    print("Test cospomdp_basic search (prior={})".format(prior))
    detectables = [target]
    if other is not None:
        detectables.append(other)

    agent_init_inputs = ['grid_map', 'camera_pose']
    if prior == "informed":
        agent_init_inputs.append('groundtruth_prior')

    detector_specs = {
        target: ("fan-far", dict(fov=90, min_range=1, mean_range=target_range),
                 (target_accuracy, target_false_pos, 0.5))
    }
    corr_specs = {}
    if other is not None:
        corr_specs[(target, other)] = (around, dict(d=dist))
        detector_specs[other] =\
            ("fan-far", dict(fov=90, min_range=1, mean_range=other_range),
             (other_accuracy, other_false_pos, 0.5))

    args = TaskArgs(detectables=detectables,
                    scene='FloorPlan1',
                    target=target,
                    agent_class="ThorObjectSearchBasicCosAgent",
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    use_vision_detector=use_vision_detector,
                    agent_init_inputs=agent_init_inputs,
                    agent_detector_specs=detector_specs,
                    corr_specs=corr_specs)
    config = make_config(args)

    config["agent_config"]["solver"] = "pomdp_py.POUCT"
    config["agent_config"]["solver_args"] = dict(max_depth=max_depth,
                                                 num_sims=num_sims,
                                                 discount_factor=discount_factor,
                                                 exploration_const=exploration_const,
                                                 show_progress=show_progress)
    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    if save:
        method = "#corr" if other is not None else "#target-only"
        config['task_config']['detector_config']['plot_detections'] = True
        config['save_path'] = os.path.join(f"./test-{scene}-{target}-cosagent-basic{method}", "vis")
        config['save_opts'] = {'gif': True,
                               'duration': 0.25}
    trial = ThorObjectSearchTrial("test_cosagent-basic", config, verbose=True)
    print("Trial created")
    trial.run(step_act_cb=step_act_cb,
              step_act_args=step_act_args,
              step_update_cb=step_update_cb)


if __name__ == "__main__":
    _test_basic_search('PepperShaker', 'StoveBurner', save=True,
                        max_steps=45)
