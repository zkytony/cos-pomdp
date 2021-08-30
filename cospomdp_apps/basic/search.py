# This is a toy domain for 2D COS-POMDP
import pomdp_py
from cospomdp.models.reward_model import ObjectSearchRewardModel
from .visual import BasicViz2D
from .basic_env import BasicEnv2D
from .parser import create_instance

def solve(worldstr, nsteps=50, solver="pomdp_py.POUCT", solver_args={}):
    agent, objlocs, colors = create_instance(worldstr)

    if solver == "pomdp_py.POUCT":
        planner = pomdp_py.POUCT(**solver_args,
                                 rollout_policy=agent.policy_model)

    env = BasicEnv2D(agent.belief.mpe().s(agent.robot_id),
                     objlocs,
                     agent.target_id,
                     agent.transition_model.robot_trans_model.reachable_positions,
                     agent.reward_model)

    is_search_task = isinstance(agent.reward_model, ObjectSearchRewardModel)

    viz = BasicViz2D(region=agent.search_region)
    viz.on_init()
    viz.visualize(agent, objlocs, colors, draw_fov=True,
                  draw_belief=is_search_task)

    for i in range(nsteps):
        action = planner.plan(agent)
        observation, reward = env.execute(action, agent.observation_model)

        planner_info = ""
        if isinstance(planner, pomdp_py.POUCT):
            planner_info += "   NumSims: %d" % planner.last_num_sims
            planner_info += "   PlanTime: %.5f" % planner.last_planning_time
            pomdp_py.utils.TreeDebugger(agent.tree).mbp
        print(f"Step {i} | a: {action}   r: {reward}    z: {observation}   | {planner_info}")

        agent.update(action, observation)
        planner.update(agent, action, observation)
        viz.visualize(agent, objlocs, colors, draw_fov=True,
                      draw_belief=is_search_task)
        if action.name == "done":
            break


WORLD =\
"""
### map
R........
.x..xG...
.x.Tx....

### robotconfig
th: 0

### corr
T around G: d=2

### detectors
T: fan-nofp | fov=45, min_range=0, max_range=2 | (0.6, 0.1)
G: fan-nofp | fov=45, min_range=0, max_range=3 | (0.8, 0.1)

### goal
find: T, 2.0

### colors
T: [0, 22, 120]
G: [0, 210, 20]

### END
"""

if __name__ == "__main__":
    solve(WORLD, nsteps=50,
          solver="pomdp_py.POUCT",
          solver_args=dict(max_depth=15,
                           planning_time=1.,
                           discount_factor=0.95,
                           exploration_const=100))
