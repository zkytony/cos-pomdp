import time
import thortils
from cosp.utils.math import euclidean_dist
from cosp.thor.agent import (ThorObjectSearchCOSPOMDP,
                             HighLevelSearchRegion,
                             HighLevelCorrelationDist)
from cosp.thor.object_search import ThorObjectSearch
from cosp.thor import constants
from cosp.planning.hierarchical import HierarchicalPlanningAgent

CORR_MATRIX = {
    ("Apple", "CounterTop"): 0.7,
    ("Apple", "Bread"): 0.8,
    ("Bread", "CounterTop"): 0.7,
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
    corr = CORR_MATRIX[(target_class, objclass)]
    distance = euclidean_dist(target_pos, object_pos)
    if corr > 0:
        return (1.0 - corr) / (distance + 0.2)
    else:
        # 7.5 is a distance threshol
        return (1.0 - abs(corr)) / (-distance - 7.5)

def test_create():
    scene = "FloorPlan1"
    target_class = "Apple"
    thor_config = {**constants.CONFIG, **{"scene": scene}}
    task_config = {
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
            "res": 30
        }
    }

    controller = thortils.launch_controller(thor_config)
    search_region = HighLevelSearchRegion(thortils.thor_reachable_positions(controller))
    x, _, z = thortils.thor_agent_position(controller, as_tuple=True)
    init_robot_pos = (x, z)  # high-level position

    detection_config = {
        "CounterTop": 0.7,
        "Apple": 0.5,
        "Bread": 0.6
    }

    corr_dists = {
        objclass: HighLevelCorrelationDist(objclass, target_class, search_region, corr_func)
        for objclass in detection_config
        if objclass != target_class}

    planning_config = {
        "max_depth": 10,
        "discount_factor": 0.95,
        "num_sims": 100,
        "exploration_const": constants.TOS_REWARD_HI - constants.TOS_REWARD_LO
    }

    print("Creating High Level POMDP...")
    high_level_pomdp = ThorObjectSearchCOSPOMDP(task_config,
                                                search_region,
                                                init_robot_pos,
                                                detection_config,
                                                corr_dists,
                                                planning_config)

    # Create a task environment
    task_env = ThorObjectSearch(controller, task_config)
    agent = HierarchicalPlanningAgent(high_level_pomdp)

    print("Planning one step...")
    _start_time = time.time()
    print(high_level_pomdp.plan_step())

    viz = task_env.visualizer(**task_config["viz_config"])
    viz.visualize(task_env, agent)

    print("Took {:3f}s".format(time.time() - _start_time))

if __name__ == "__main__":
    test_create()
