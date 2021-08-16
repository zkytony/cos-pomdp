# This is a toy domain for 2D COS-POMDP
import pomdp_py
import numpy as np
from cospomdp.utils.visual import Visualizer
from cospomdp.utils.world import create_instance
from cospomdp.domain.state import CosState2D, ObjectState2D
from cospomdp.models.transition_model import FullTransitionModel2D, RobotTransition2D

class ToyViz(Visualizer):
    def visualize(self, robot_state, target_belief, objlocs, colors):
        img = self._make_gridworld_image(self._res)
        x, y, th = robot_state["pose"]
        for objid in objlocs:
            img = self.highlight(img, [objlocs[objid]], colors[objid])
        target_id = target_belief.mpe().id
        img = self.draw_object_belief(img, target_belief, colors[target_id] + [250])
        img = self.draw_robot(img, x, y, th, (255, 20, 20))
        self.show_img(img)

class ToyEnv(pomdp_py.Environment):
    def __init__(self, init_robot_state, objlocs, target_id,
                 reachable_positions, reward_model):
        objstates = {objid: ObjectState2D(objid, objid, objlocs[objid])
                     for objid in objlocs}
        init_state = CosState2D({**{init_robot_state.id:init_robot_state},
                                 **objstates})
        robot_trans_model = RobotTransition2D(init_robot_state.id,
                                              reachable_positions)
        transition_model = FullTransitionModel2D(robot_trans_model)
        super().__init__(init_state, transition_model=transition_model,
                         reward_model=reward_model)


WORLD =\
"""
### map
R....
.x.Tx
.xG.x

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
    agent, objlocs, colors = create_instance(WORLD)
    planner = pomdp_py.POUCT(max_depth=10, discount_factor=0.95,
                             planning_time=1., exploration_const=100,
                             rollout_policy=agent.policy_model)

    env = ToyEnv(agent.belief.mpe().s(agent.robot_id),
                 objlocs,
                 agent.target_id,
                 agent.transition_model.robot_trans_model.reachable_positions,
                 agent.reward_model)

    viz = ToyViz(region=agent.search_region)
    viz.on_init()

    for i in range(20):
        action = planner.plan(agent)
        observation, reward = env.execute(action, agent.observation_model)
        print(f"Step {i} | a: {action}   r: {reward}    z: {observation}")

        agent.update(action, observation)
        planner.update(agent, action, observation)

        robot_state = agent.belief.mpe().s(agent.robot_id)
        target_belief = agent.belief.b(agent.target_id)
        viz.visualize(robot_state, target_belief, objlocs, colors)

        if action.name == "done":
            break
