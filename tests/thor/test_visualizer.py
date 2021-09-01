import thortils
import thortils.constants as constants
from thortils.utils import getch
import argparse
import time

from cospomdp_apps.thor.visual import GridMapVisualizer

def _test_visualizer(args=None, sleep=15):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Keyboard control of agent in ai2thor")
        parser.add_argument("-s", "--scene",
                            type=str, help="scene. E.g. FloorPlan1",
                            default="FloorPlan1")
        args = parser.parse_args()

    print("Testing visualizer")

    controller = thortils.launch_controller({**constants.CONFIG, **{"scene": args.scene}})

    reachable_positions = thortils.thor_reachable_positions(controller)
    grid_map = thortils.proper_convert_scene_to_grid_map(controller, constants.GRID_SIZE)
    # convert_scene_to_grid_map(controller, args.scene, constants.GRID_SIZE)
    print(thortils.thor_agent_pose(controller))

    print("Start at (2,12)")
    viz = GridMapVisualizer(grid_map=grid_map)
    img = viz.render()
    img = viz.highlight(img, [(2,12)], color=(128,128,128))
    viz.show_img(img)

    pos, rot = thortils.thor_agent_pose(controller)

    print("Translate -x by 0.25")
    img = viz.highlight(img, [(pos['x'], pos['z'])], color=(10,150,150), thor=True)
    img = viz.highlight(img, [(pos['x']-0.25, pos['z'])], color=(10,180,180), thor=True)
    thortils.thor_teleport2d(controller, (pos['x']-0.25, pos['z'], rot['y']))
    viz.show_img(img)
    time.sleep(0.8)

    print("Translate +z by 0.25")
    img = viz.highlight(img, [(pos['x']-0.25, pos['z']+0.25)], color=(10,180,180), thor=True)
    thortils.thor_teleport2d(controller, (pos['x']-0.25, pos['z']+0.25, rot['y']))
    viz.show_img(img)
    time.sleep(0.8)

    # What about angle?
    print("rotate to yaw = 0")
    gx, gy, gth = grid_map.to_grid_pose(pos['x'], pos['z'], 0)
    img = viz.draw_robot(img, gx, gy, gth, color=(200, 140, 194))
    thortils.thor_teleport2d(controller, (pos['x'], pos['z'], 0))
    viz.show_img(img)
    time.sleep(0.8)

    print("rotate to yaw = 90")
    gx, gy, gth = grid_map.to_grid_pose(pos['x']+0.5, pos['z']+0.5, 90)
    img = viz.draw_robot(img, gx, gy, gth, color=(200, 140, 194))
    thortils.thor_teleport2d(controller, (pos['x']+0.5, pos['z']+0.5, 90))
    viz.show_img(img)

    time.sleep(sleep)
    controller.stop()
    viz.on_cleanup()


if __name__ == "__main__":
    _test_visualizer()
