from cosp.thor.trial import ThorTrial
from cosp.thor import constants


def build_object_search_trial(target, task_type, max_steps=100):
    task_config = {
        "task_type": task_type,
        "target": target,
    }

    thor_config = {**constants.CONFIG, **{"scene": "FloorPlan1"}}
    config = {
        "thor": thor_config,
        "max_steps": max_steps,
        "task_env": "ThorObjectSearch",
        "task_env_config": {**task_config, **{"goal_distance": constants.GOAL_DISTANCE}},
        "agent_class": "ThorObjectSearchOptimalAgent",
        "agent_config": task_config
    }

    trial = ThorTrial("test_optimal", config, verbose=True)
    return trial
