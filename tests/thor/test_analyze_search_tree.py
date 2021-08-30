from pomdp_py.utils import TreeDebugger
from thortils.utils.colors import linear_color_gradient
from test_cosagent_basic_search import _test_basic_search
from cospomdp_apps.basic.action import Move2D
from pprint import pprint
import time
from tqdm import tqdm
import random

def _get_robot_traj(dd, node, cos_agent):
    """
    Given the tree debugger, and a node on the tree (VNode),
    find the path from root node to the given node, and recover
    the robot's trajectory along this path.
    """
    mpe_state = cos_agent.belief.mpe()
    robot_states = [mpe_state.s(cos_agent.robot_id)]
    actions = []

    path = dd.path_to(node)
    state = mpe_state
    for edge in path:
        if isinstance(edge, Move2D):
            next_state = cos_agent.transition_model.sample(state, edge)
            robot_states.append(next_state.s(cos_agent.robot_id))
            actions.append(edge)
            state = next_state
    return robot_states, actions

def _draw_robot_traj(viz, task_env, agent, robot_states, actions, animate=True):
    img = viz.render(task_env, agent, len(agent.cos_agent.history))
    colors = linear_color_gradient((219, 171, 13), (13, 27, 219), len(robot_states))
    for i, sr in enumerate(robot_states):
        if i > 0 and i < len(actions):
            print(i, actions[i-1])
        x, y, th = sr['pose']
        img = viz.draw_robot(img, x, y, th, color=colors[i], thickness=3)
        if animate:
            viz.show_img(img)
            time.sleep(0.2)
    if not animate:
        viz.show_img(img)

def interactive_show_robot_trajs(viz, task_env, agent, depth=None, num=100):
    """num: Number of trajectories to be able to iterate trough"""
    dd = TreeDebugger(agent.cos_agent.tree)
    if depth is None:
        nodes = dd.leaf  # get all nodes on the last layer
    else:
        nodes = dd.l(depth)

    random.shuffle(nodes)
    outputs = []
    for i, node in enumerate(tqdm(nodes[:num])):
        traj, actions = _get_robot_traj(dd, node, agent.cos_agent)
        outputs.append((traj, actions))

    for i, tup in enumerate(reversed(sorted(outputs, key=lambda t: len(t[0])))):
        traj, actions = tup
        print("Path {} out of {}; Length = {} ".format(i, len(nodes), len(traj)))
        _draw_robot_traj(viz, task_env, agent, traj, actions)
        cont = input("Continue? [y] ").startswith("y")
        if not cont:
            return

def step_act_cb(task_env, agent, **kwargs):
    viz = kwargs.get("viz")
    interactive_show_robot_trajs(viz, task_env, agent, depth=None)

if __name__ == "__main__":
    _test_basic_search('Bowl', 'Book', prior='informed',
                       step_act_cb=step_act_cb,
                       num_sims=10000,
                       show_progress=True,
                       max_depth=30,
                       exploration_const=50)
