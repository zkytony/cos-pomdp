from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor.common import TaskArgs, make_config

def _test_random_agent(target,
                       scene="FloorPlan1",
                       target_range=5,
                       target_quality=(0.7, 0.05),
                       max_steps=100):
    print("Test cospomdp_random agent")
    agent_init_inputs = ["grid_map"]
    args = TaskArgs(detectables=[target],
                    scene='FloorPlan1',
                    target=target,
                    agent_class="ThorObjectSearchRandomAgent",
                    task_env="ThorObjectSearch",
                    max_steps=max_steps,
                    agent_init_inputs=agent_init_inputs)
    config = make_config(args)

    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_cosagent-basic", config, verbose=True)
    print("Trial created")
    trial.run()

if __name__ == "__main__":
    _test_random_agent('Bowl')
