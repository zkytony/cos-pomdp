from cospomdp.utils.corr_funcs import around, apart
from cospomdp_apps.thor.agent.cospomdp_complete\
    import ThorObjectSearchCompleteCosAgent
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor.agent.components.topo_map\
    import TopoMap, draw_edge, draw_topo, mark_cell
from pomdp_py.utils import TreeDebugger

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
GRID_MAP_DATASET_PATH = os.path.join(ABS_PATH, "../../data/thor/grid_maps")

def step_act_cb(task_env, agent, **kwargs):
    # viz = kwargs.get("viz")
    pass

    # img = viz.render(task_env, agent, len(agent.cos_agent.history))
    # img = draw_topo(img, agent.topo_map, viz._res, draw_grid_path=True)
    # viz.show_img(img)
    # import pdb; pdb.set_trace()

def _test_complete_search(target,
                          other,
                          scene="FloorPlan1",
                          prior='uniform',
                          dist=3,
                          target_range=6,
                          other_range=8,
                          target_accuracy=0.7,
                          other_accuracy=0.8,
                          target_false_pos=None,
                          other_false_pos=None,
                          use_vision_detector=False,
                          max_depth=30,
                          num_sims=500,
                          max_steps=100,
                          num_place_samples=20,
                          discount_factor=0.95,
                          exploration_const=100,
                          local_search_type="basic",
                          local_search_params={},
                          show_progress=True,
                          step_act_cb=None,
                          step_act_args={},
                          step_update_cb=None,
                          setup_only=False):
    print("Test cospomdp_complete search (prior={})".format(prior))
    print("Target object: {}".format(target))
    print("Other object: {}".format(other))
    detectables = [target]
    if other is not None:
        detectables.append(other)

    agent_init_inputs = ['grid_map', 'agent_pose']
    if prior == "informed":
        agent_init_inputs.append('groundtruth_prior')
    if local_search_type == "3d":
        agent_init_inputs.append('height_range')

    if target_false_pos is not None:
        quality = (target_accuracy, target_false_pos, 0.5)
        target_detector = ("fan-far", dict(fov=90, min_range=1, mean_range=target_range), quality)
    else:
        target_detector = ("fan-nofp", dict(fov=90, min_range=1, max_range=target_range), (target_accuracy, 0.1))

    detector_specs = {
        target: target_detector
    }
    corr_specs = {}
    if other is not None:
        corr_specs[(target, other)] = (around, dict(d=dist))

        if other_false_pos is not None:
            quality = (other_accuracy, other_false_pos, 0.5)
            other_detector = ("fan-simplefp", dict(fov=90, min_range=1, max_range=other_range), quality)
        else:
            other_detector = ("fan-nofp", dict(fov=90, min_range=1, max_range=other_range), (other_accuracy, 0.1))
        detector_specs[other] = other_detector

    args = TaskArgs(detectables=detectables,
                    scene=scene,
                    target=target,
                    agent_class="ThorObjectSearchCompleteCosAgent",
                    task_env="ThorObjectSearch",
                    agent_init_inputs=agent_init_inputs,
                    use_vision_detector=use_vision_detector,
                    plot_detections=True,
                    agent_detector_specs=detector_specs,
                    corr_specs=corr_specs)
    config = make_config(args)

    config["agent_config"]["num_place_samples"] = num_place_samples

    config["agent_config"]["solver"] = "pomdp_py.POUCT"
    config["agent_config"]["solver_args"] = dict(max_depth=max_depth,
                                                 num_sims=num_sims,
                                                 discount_factor=discount_factor,
                                                 exploration_const=exploration_const,
                                                 show_progress=show_progress)
    config["agent_config"]["local_search_type"] = local_search_type
    config["agent_config"]["local_search_params"] = local_search_params

    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_cosagent-complete", config, verbose=True)
    print("Trial created")
    if setup_only:
        return trial.setup()
    else:
        trial.run(step_act_cb=step_act_cb,
                  step_act_args=step_act_args,
                  step_update_cb=step_update_cb,
                  logging=True)

if __name__ == "__main__":
    _test_complete_search("SaltShaker", "StoveBurner",
                          scene="FloorPlan1",
                          step_act_cb=step_act_cb,
                          num_sims=100,
                          target_false_pos=0.15,
                          other_false_pos=0.1,
                          use_vision_detector=False,
                          local_search_type="3d",
                          local_search_params={"pouct": {"num_sims": 200,
                                                         "max_depth": 30,
                                                         "discount_factor": 0.95,
                                                         "exploration_const": 100,
                                                         "show_progress": True}})
