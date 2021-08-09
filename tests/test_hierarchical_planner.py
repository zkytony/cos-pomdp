# Test the hierarchical planner as a whole
# The goal here is to make the agent search for objects.


import time
import thortils
from test_utils import corr_func
from cosp.thor.agent import (ThorObjectSearchCOSPOMDP,
                             HighLevelSearchRegion,
                             HighLevelCorrelationDist)
from cosp.thor.object_search import ThorObjectSearch
from cosp.thor.trial import ThorObjectSearchTrial
from cosp.thor import constants
from cosp.planning.hierarchical import HierarchicalPlanningAgent

def test_create():
    robot_id = "robot0"
    scene = "FloorPlan1"
    target_class = "Apple"
    thor_config = {**constants.CONFIG, **{"scene": scene}}
    task_config = {
        "robot_id": robot_id,
        "task_type": "class",
        "target": target_class,
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

    detection_config = {
        "CounterTop": 0.7,
        "Apple": 0.5,
        "Bread": 0.6
    }

    planning_config = {
        "max_depth": 10,
        "discount_factor": 0.95,
        "num_sims": 100,
        "exploration_const": constants.TOS_REWARD_HI - constants.TOS_REWARD_LO
    }

    agent_config = {"task_config": task_config,
                    "detection_config": detection_config,
                    "corr_func": corr_func,
                    "planning_config": planning_config}

    config = {
        "thor": thor_config,
        "max_steps": 100,
        "task_env": "ThorObjectSearch",
        "task_env_config": {"task_config": task_config},
        "agent_class": "ThorObjectSearchCOSPOMDPAgent",
        "agent_config": agent_config
    }
    trial = ThorObjectSearchTrial("test_hierarchical", config)
    return trial.run(logging=True)


if __name__ == "__main__":
    test_create()
