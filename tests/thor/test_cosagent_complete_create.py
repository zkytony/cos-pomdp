from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.agent.cospomdp_complete\
    import _sample_places, ThorObjectSearchCompleteCosAgent
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor.agent.components.topo_map\
    import TopoMap, draw_edge, draw_topo, mark_cell
from pomdp_py.utils import TreeDebugger

def step_act_cb(task_env, agent, **kwargs):
    viz = kwargs.get("viz")

    img = viz.render(task_env, agent, len(agent.cos_agent.history))
    img = draw_topo(img, agent.lll, viz._res, draw_grid_path=True)
    viz.show_img(img)
    import pdb; pdb.set_trace()


def _test_sampling_topo_map():
    args = TaskArgs(detectables={"Apple", "CounterTop", "Bread"},
                    scene='FloorPlan1',
                    target="Apple",
                    agent_class="ThorObjectSearchCompleteCosAgent",
                    task_env="ThorObjectSearch",
                    agent_init_inputs=['grid_map', 'agent_pose'])
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
    config["agent_config"]["num_place_samples"] = 10
    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_cosagent", config)
    print("Trial created")
    trial.run(step_act_cb=step_act_cb)

if __name__ == "__main__":
    _test_sampling_topo_map()
