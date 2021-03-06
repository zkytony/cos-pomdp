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

def _draw_robot_traj(viz, task_env, agent, robot_states, actions, img=None, show_each=False):
    if img is None:
        img = viz.render(task_env, agent, len(agent.cos_agent.history))
    colors = linear_color_gradient((219, 171, 13), (13, 27, 219), len(robot_states))
    for i, sr in enumerate(robot_states):
        if i > 0 and i < len(actions):
            print(i, actions[i-1])
        x, y, th = sr['pose']
        img = viz.draw_robot(img, x, y, th, color=colors[i], thickness=3)
        if show_each:
            viz.show_img(img)
            time.sleep(0.2)
    return img

def interactive_show_robot_trajs(viz, task_env, agent, depth=None, num_trajs=100, show_each=False, interactive=True):
    """num: Number of trajectories to be able to iterate trough"""
    dd = TreeDebugger(agent.cos_agent.tree)
    if depth is None:
        nodes = dd.leaf  # get all nodes on the last layer
    else:
        nodes = dd.l(depth)

    random.shuffle(nodes)
    outputs = []
    for i, node in enumerate(tqdm(nodes[:num_trajs])):
        traj, actions = _get_robot_traj(dd, node, agent.cos_agent)
        outputs.append((traj, actions))

    img = None
    for i, tup in enumerate(reversed(sorted(outputs, key=lambda t: len(t[0])))):
        traj, actions = tup
        print("Path {} out of {}; Length = {} ".format(i, len(nodes), len(traj)))
        img = _draw_robot_traj(viz, task_env, agent, traj, actions, img=img, show_each=show_each)
        if show_each:
            cont = input("Continue? [y] ").startswith("y")
            if not cont:
                return
    if not show_each:
        viz.show_img(img)
        if interactive:
            cont = input("Continue? [y] ").startswith("y")
            if not cont:
                exit(0)

def step_act_cb(task_env, agent, **kwargs):
    viz = kwargs.get("viz")
    interactive_show_robot_trajs(viz, task_env, agent,
                                 num_trajs=kwargs.get("num_trajs", 30),
                                 depth=kwargs.get("depth", None),
                                 interactive=kwargs.get('interactive', False))

def _test_analyze_cosagent_basic_search_tree(target, other,
                                             num_trajs=30,
                                             depth=None,
                                             interactive=True,
                                             **kwargs):
    _test_basic_search(target, other, prior='informed',
                       step_act_cb=step_act_cb,
                       step_act_args={'num_trajs': num_trajs,
                                      'interactive': interactive},
                       **kwargs)

if __name__ == "__main__":
    _test_analyze_cosagent_basic_search_tree(
        'Bowl', 'Book', num_sims=10000,
        show_progress=True, max_depth=30, exploration_const=0)
