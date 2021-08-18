import thortils

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial


def _test_basic_search():
    prior = 'uniform'
    args = TaskArgs(detectables={"Bowl", "Book"},
                    scene='FloorPlan1',
                    target="Bowl",
                    agent_class="ThorObjectSearchCosAgent",
                    task_env="ThorObjectSearch",
                    prior=prior)
    config = make_config(args)
    config["agent_config"]["prior"] = prior
    config["agent_config"]["corr_specs"] = {
        ("Bowl", "Book"): (around, dict(d=3))
    }
    config["agent_config"]["detector_specs"] = {
        "Bowl": ("fan-nofp", dict(fov=90, min_range=1, max_range=5), (0.7, 0.1)),
        "Book": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1))
    }
    config["agent_config"]["solver"] = "pomdp_py.POUCT"
    config["agent_config"]["solver_args"] = dict(max_depth=30,
                                                 num_sims=500,
                                                 discount_factor=0.95,
                                                 exploration_const=100)
    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_cosagent", config, verbose=True)
    print("Trial created")
    trial.run()


if __name__ == "__main__":
    _test_basic_search()
