from cospomdp.domain.action import Move2D
from pomdp_py.utils import TreeDebugger
from test_cosagent_basic_search import _test_basic_search

def _get_robot_traj(dd, node, cos_agent):
    """
    Given the tree debugger, and a node on the tree (VNode),
    find the path from root node to the given node, and recover
    the robot's trajectory along this path.
    """
    mpe_state = cos_agent.belief.mpe()
    robot_states = [mpe_state.s(cos_agent.robot_id)]

    path = dd.path_to(node)
    state = mpe_state
    for edge in path:
        if isinstance(edge, Move2D):
            next_state = cos_agent.transition_model.sample(state, edge)
            robot_states.append(next_state.s(cos_agent.robot_id))
    return robot_states

def _draw_robot_traj(viz, task_env, agent, robot_states):
    img = viz.render(task_env, agent, len(agent.cos_agent.history))
    for sr in robot_states:
        x, y, th = sr['pose']
        img = viz.draw_robot(img, x, y, th, color=(0, 0, 255), thickness=1)
    viz.show_img(img)

def interactive_show_robot_trajs(viz, task_env, agent, depth=None):
    dd = TreeDebugger(agent.cos_agent.tree)
    if depth is None:
        nodes = dd.l(dd.d)  # get all nodes on the last layer
    else:
        nodes = dd.l(depth)

    for i, node in enumerate(nodes):
        print("Path {} out of {}".format(i, len(nodes)))
        traj = _get_robot_traj(dd, node, agent.cos_agent)
        _draw_robot_traj(viz, task_env, agent, traj)
        cont = input("Continue? [y]").startswith("y")
        if not cont:
            return

def step_act_cb(task_env, agent, **kwargs):
    viz = kwargs.get("viz")
    interactive_show_robot_trajs(viz, task_env, agent, depth=None)

if __name__ == "__main__":
    _test_basic_search('Bowl', 'Book', prior='informed',
                       step_act_cb=step_act_cb,
                       num_sims=200,
                       show_progress=True,
                       max_depth=30)
