####################
# DYSFUNCT
####################
# Test the hierarchical planner as a whole
# The goal here is to make the agent search for objects.


import time
import thortils
from cosp.thor.trial import ThorObjectSearchTrial
from cosp.utils.math import euclidean_dist
from cosp.thor import constants

# Hard coded correlation
CORR_MATRIX = {
    ("Apple", "CounterTop"): 0.7,
    ("Apple", "Bread"): 0.8,
    ("Apple", "Fridge"): -1
}
for k in list(CORR_MATRIX.keys()):
    CORR_MATRIX[tuple(reversed(k))] = CORR_MATRIX[k]

def corr_func(target_pos, object_pos,
              target_class, objclass):
    """
    Returns a float value to essentially mean
    Pr(Si = object_pos | Starget = target_pos)
    """
    # This is a rather arbitrary function for now.
    if target_class == objclass:
        corr = 1.0
    else:
        corr = CORR_MATRIX[(target_class, objclass)]
    distance = euclidean_dist(target_pos, object_pos)
    if corr > 0:
        return distance <= 2.0
    else:
        return distance >= 2.0


def test_create():
    robot_id = "robot0"
    scene = "FloorPlan1"
    target_class = "Apple"

    detectables = [("Apple", "FanModelNoFP", [0.7, 0.1])]
                   # ("CounterTop", "FanModelNoFP", [0.9, 0.1]),
                   # ("Bread", "FanModelNoFP", [0.7, 0.1]),
                   # ("Fridge", "FanModelNoFP", [0.9, 0.1])]
    detector_config = {}
    for cls, model_type, quality_params in detectables:
        if model_type == "FanModelNoFP":
            fan_params = {
                "min_range": 1,
                "max_range": 4,
                "fov": constants.FOV
            }
            detector_config[cls] =\
                dict(type=model_type,
                     params=dict(
                         objclass=target_class,
                         fan_params=fan_params,
                         quality_params=quality_params,
                         round_to=constants.GRID_SIZE))

    thor_config = {**constants.CONFIG, **{"scene": scene}}
    task_config = {
        "robot_id": robot_id,
        "task_type": "class",
        "target": target_class,
        "detectables": set(detector_config.keys()),
        "nav_config": {
            "goal_distance": constants.GOAL_DISTANCE,
            "v_angles": constants.V_ANGLES,
            "h_angles": constants.H_ANGLES,
            "diagonal_ok": constants.DIAG_MOVE,
            "movement_params": thor_config["MOVEMENT_PARAMS"]
        },
        "discount_factor": 0.99,
        "visualize": True,
        "viz_config": {
            "res": 30,
            "colors": {
                robot_id: [255, 100, 255],
                target_class: [100, 100, 255, 128]
            }
        }
    }


    planning_configs = {
        "max_depth": 10,
        "discount_factor": 0.95,
        "num_sims": 100,
        "exploration_const": constants.TOS_REWARD_HI - constants.TOS_REWARD_LO
    }

    agent_config = {"task_config": task_config,
                    "detector_config": detector_config,
                    "corr_func": corr_func,
                    "planning_config": planning_configs}

    config = {
        "thor": thor_config,
        "max_steps": 100,
        "task_env": "ThorObjectSearch",
        "task_env_config": {"task_config": task_config},
        "agent_class": "ThorObjectSearchCOSPOMDPAgent",
        "agent_config": agent_config,
        "visualize": True,
        "viz_config": {
            "res": 30,
            "colors": {
                robot_id: [255, 100, 255],
                target_class: [100, 100, 255, 128]
            }
        }
    }
    trial = ThorObjectSearchTrial("test_hierarchical", config, verbose=True)
    return trial.run(logging=True)


if __name__ == "__main__":
    test_create()
