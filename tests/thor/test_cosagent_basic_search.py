import thortils

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial


def _test_basic_search(target,
                       other,
                       prior='uniform',
                       scene="FloorPlan1",
                       dist=3,
                       target_range=5,
                       other_range=6,
                       target_accuracy=0.7,
                       other_accuracy=0.8,
                       max_depth=30,
                       num_sims=500,
                       discount_factor=0.95,
                       show_progress=True,
                       step_act_cb=None,
                       step_update_cb=None):
    args = TaskArgs(detectables=[target, other],
                    scene='FloorPlan1',
                    target=target,
                    agent_class="ThorObjectSearchCosAgent",
                    task_env="ThorObjectSearch",
                    prior=prior)
    config = make_config(args)
    config["agent_config"]["prior"] = prior
    config["agent_config"]["corr_specs"] = {
        (target, other): (around, dict(d=dist))
    }
    config["agent_config"]["detector_specs"] = {
        target: ("fan-nofp", dict(fov=90, min_range=1, max_range=target_range), (target_accuracy, 0.1)),
        other: ("fan-nofp", dict(fov=90, min_range=1, max_range=other_range), (other_accuracy, 0.1))
    }
    config["agent_config"]["solver"] = "pomdp_py.POUCT"
    config["agent_config"]["solver_args"] = dict(max_depth=max_depth,
                                                 num_sims=num_sims,
                                                 discount_factor=discount_factor,
                                                 exploration_const=100,
                                                 show_progress=show_progress)
    config["visualize"] = True
    config["viz_config"] = {
        'res': 30
    }
    trial = ThorObjectSearchTrial("test_cosagent", config, verbose=True)
    print("Trial created")
    trial.run(step_act_cb=step_act_cb,
              step_update_cb=step_update_cb)


if __name__ == "__main__":
    _test_basic_search('Bowl', 'Book')
