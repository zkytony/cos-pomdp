# This is a toy domain for 2D COS-POMDP
import pomdp_py
import numpy as np
from cospomdp.utils.visual import Visualizer
from cospomdp.utils.world import create_instance

class ToyViz(Visualizer):
    def visualize(self, robot_state, target_belief):
        img = self._make_gridworld_image(self._res)
        x, y, th = robot_state["pose"]
        img = self.draw_robot(img, x, y, th, (255, 20, 20))
        img = self.draw_object_belief(img, target_belief, (20, 20, 255))
        self.show_img(img)

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

### END
"""


if __name__ == "__main__":
    agent, objlocs = create_instance(WORLD)
    planner = pomdp_py.POUCT(max_depth=10, discount_factor=0.95,
                             planning_time=1., exploration_const=100,
                             rollout_policy=agent.policy_model)
    action = planner.plan(agent)
    print(action)
