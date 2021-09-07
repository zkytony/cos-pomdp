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
                       target_quality=(0.7, 0.05),
                       max_steps=100):
    print("Test cospomdp_random agent")
    agent_init_inputs = ["grid_map", "agent_pose"]
    args = TaskArgs(detectables=[target],
                    scene='FloorPlan1',
                    target=target,
                    agent_class="ThorObjectSearchGreedyNbvAgent",
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    agent_init_inputs=agent_init_inputs)
    config = make_config(args)

    config["agent_config"]["corr_specs"] = {}
    config["agent_config"]["detector_specs"] = {
        target: ("fan-nofp", dict(fov=90, min_range=1, max_range=target_range), (target_accuracy, 0.1))
    }
    if other is not None:
        config["agent_config"]["corr_specs"][(target, other)] = (around, dict(d=dist))
        config["agent_config"]["detector_specs"][other] =\
            ("fan-nofp", dict(fov=90, min_range=1, max_range=other_range), (other_accuracy, 0.1))


    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_greedy", config, verbose=True)
    print("Trial created")
    trial.run()

if __name__ == "__main__":
    _test_greedy_agent('Bowl')
