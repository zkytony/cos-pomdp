from cosp.models.transition import RobotTransition2D
from cosp.models.state import JointState2D, ObjectState2D
from cosp.models.action import Move
from cosp.utils.math import to_rad
from cosp.utils.plotting import plot_pose

import matplotlib.pyplot as plt

def test():
    Trobot = RobotTransition2D(grid_size=0.25)

    actions = {"forward": (1.0, 0.0),
               "left": (0.0, -45.0),
               "right": (0.0, 45.0),
               "back": (-1.0, 0.0)}

    init_pose = (2, 5, 90)
    state = JointState2D("robot", None,
                         {"robot": ObjectState2D("robot", dict(pose=init_pose))})

    path = ["forward",
            "left",
            "left",
            "forward",
            "right",
            "right",
            "forward",
            "right",
            "right",
            "back"]

    poses = [state.robot_state["pose"]]
    for a in path:
        srobot = Trobot.sample(state, Move(a, actions[a]))
        state = JointState2D("robot", None,
                             {"robot": srobot})
        poses.append(state.robot_state["pose"])

    w, l = 10, 10
    fig, ax = plt.subplots()
    ax.set_xlim(0, w)
    ax.set_ylim(0, l)
    ax.set_aspect('equal')
    for i, robot_pose in enumerate(poses):
        print(robot_pose)
        plot_pose(ax, robot_pose[0:2], to_rad(robot_pose[2]), length=0.2, head=0.1, d="+y")
        if i == 0:
            plt.show(block=False)
            ax.set_title("--")
        else:
            ax.set_title(path[i-1])
        plt.pause(0.5)
    plt.close()

if __name__ == "__main__":
    test()
