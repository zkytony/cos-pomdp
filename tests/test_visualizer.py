# COPIED FROM thortils.scripts.kbcontrol

# Keyboard control of Ai2Thor

import thortils
import thortils.constants as constants
from thortils.utils import getch
import argparse
from cosp.thor.visual import ThorObjectSearchViz


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard control of agent in ai2thor")
    parser.add_argument("-s", "--scene",
                        type=str, help="scene. E.g. FloorPlan1",
                        default="FloorPlan1")
    args = parser.parse_args()
    controller = thortils.launch_controller({**constants.CONFIG, **{"scene": args.scene}})

    reachable_positions = thortils.thor_reachable_positions(controller)
    grid_map = thortils.convert_scene_to_grid_map(controller, args.scene, constants.GRID_SIZE)

    viz = ThorObjectSearchViz(grid_map=grid_map)

    controls = {
        "w": "MoveAhead",
        "a": "RotateLeft",
        "d": "RotateRight",
        "e": "LookUp",
        "c": "LookDown"
    }

    while True:
        k = getch()
        if k == "q":
            print("bye.")
            break

        if k in controls:
            action = controls[k]
            params = constants.MOVEMENT_PARAMS[action]
            controller.step(action=action, **params)

        print("Agent pose: {}".format(thortils.thor_agent_pose(controller, as_tuple=True)))

if __name__ == "__main__":
    main()
