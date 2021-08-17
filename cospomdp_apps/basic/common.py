import pomdp_py
from cospomdp.utils.visual import BasicViz2D
from cospomdp.utils.world import create_instance
from cospomdp.models.basic_env import BasicEnv2D
from cospomdp.models.reward_model import ObjectSearchRewardModel2D

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

    is_search_task = isinstance(agent.reward_model, ObjectSearchRewardModel2D)

    viz = BasicViz2D(region=agent.search_region)
    viz.on_init()
    viz.visualize(agent, objlocs, colors, draw_fov=True,
                      draw_belief=is_search_task)

    for i in range(nsteps):
        action = planner.plan(agent)
        observation, reward = env.execute(action, agent.observation_model)
        print(f"Step {i} | a: {action}   r: {reward}    z: {observation}")

        agent.update(action, observation)
        planner.update(agent, action, observation)

        viz.visualize(agent, objlocs, colors, draw_fov=True,
                      draw_belief=is_search_task)
        if action.name == "done":
            break
