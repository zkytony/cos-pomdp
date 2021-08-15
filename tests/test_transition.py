import pytest
import matplotlib.pyplot as plt
from cospomdp.models.transition_model import RobotTransition2D, CosTransitionModel2D
from cospomdp.domain.action import *
from cospomdp.domain.state import *
from cospomdp.utils.plotting import plot_pose
from cospomdp.utils.math import to_rad

@pytest.fixture
def show_plots():
    return True

@pytest.fixture
def init_srobot():
    init_pose = (2, 5, 0)
    return RobotState2D("robot", init_pose)

@pytest.fixture
def dim():
    w, l = 10, 10
    return w, l

def test_transition_joint(dim, init_srobot):
    target_id = 10
    w, l = dim
    Trobot = RobotTransition2D("robot", [(x,y)
                                         for x in range(w)
                                         for y in range(l)])
    state = CosState2D({"robot": init_srobot,
                        target_id: ObjectState2D(target_id, "target", (3,3))})
    T = CosTransitionModel2D(target_id, Trobot)
    next_state = T.sample(state, MoveAhead)
    assert next_state.s("robot")["pose"] == (3, 5, 0)
    assert next_state.s(target_id) == state.s(target_id)


def test_transition_follow_path(dim, show_plots, init_srobot):
    w, l = dim
    Trobot = RobotTransition2D("robot", [(x,y)
                                         for x in range(w)
                                         for y in range(l)])

    state = CosState2D({"robot": init_srobot})
    actions = {"forward": MoveAhead,
               "left": RotateLeft,
               "right": RotateRight}

    # Let's go in a circle
    path = ["left",
            "forward",
            "left",
            "forward",
            "right",
            "right",
            "forward",
            "right",
            "right",
            "forward",
            "right",
            "right",
            "forward",
            "left",
            "forward",
            "right"]

    poses = [state.s("robot")['pose']]
    for a in path:
        # import pdb; pdb.set_trace()
        srobot = Trobot.sample(state, actions[a])
        state = CosState2D({"robot": srobot})
        poses.append(state.s("robot")["pose"])

    if show_plots:
        fig, ax = plt.subplots()
        ax.set_xlim(0, w)
        ax.set_ylim(0, l)
        ax.set_aspect('equal')
        for i, robot_pose in enumerate(poses):
            print(robot_pose)
            plot_pose(ax, robot_pose[0:2], to_rad(robot_pose[2]), length=0.2, head=0.1, d="+x")
            if i == 0:
                plt.show(block=False)
                ax.set_title("--")
            else:
                ax.set_title(path[i-1])
            plt.pause(0.5)
        plt.close()

    assert poses[-1][:2] == poses[0][:2]
    assert poses[-1][2] == poses[0][2] + 180
