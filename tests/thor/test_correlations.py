import random
import numpy as np
from cospomdp.utils.corr_funcs import ConditionalSpatialCorrelation
import thortils as tt
import matplotlib.pyplot as plt
from tqdm import tqdm

def _test_simple_correlation_single(scene="FloorPlan1"):
    controller = tt.launch_controller({"scene":scene})
    target = (0, "Bread")
    other = (1, "Bowl")

    distances = tt.thor_distances_in_scene(controller, target[1], other[1])

    # We want distances in grid cell units
    grid_size = tt.thor_grid_size_from_controller(controller)
    distances = np.array(distances) / grid_size
    spcorr = ConditionalSpatialCorrelation(target, other, distances)

    grid_map = tt.proper_convert_scene_to_grid_map(
        controller, grid_size
    )
    rnd = random.Random(100)

    heatmap = np.zeros((grid_map.width, grid_map.length))
    loc_target = rnd.sample(grid_map.obstacles, 1)[0]
    for loc_other in grid_map.obstacles:
        score = spcorr.func(loc_target, loc_other, target[0], other[0])
        x_o, y_o = loc_other
        heatmap[x_o, y_o] = score
    plt.imshow(heatmap, cmap="hot")
    plt.scatter([loc_target[0]], [loc_target[1]], marker="o", c="cyan")
    plt.show()

def _test_simple_correlation(scene_type="kitchen"):
    """Learn correlation between two object classes and
    try out the distribution in a validation scene"""
    target = (0, "PepperShaker")
    other = (1, "StoveBurner")
    all_distances = []
    for scene in tqdm(tt.ithor_scene_names(scene_type, levels=range(1, 21))):
        controller = tt.launch_controller({"scene":scene})
        distances = tt.thor_distances_in_scene(controller, target[1], other[1])

        # We want distances in grid cell units
        grid_size = tt.thor_grid_size_from_controller(controller)
        distances = np.array(distances) / grid_size
        all_distances.extend(distances)
        controller.stop()
        break

    spcorr = ConditionalSpatialCorrelation(target, other, all_distances)

    # Try out on a validation scene
    scene = tt.ithor_scene_names(scene_type, levels=range(21, 31))[4]
    controller = tt.launch_controller({"scene":scene})

    grid_map = tt.proper_convert_scene_to_grid_map(
        controller, grid_size
    )

    loc_target = tt.thor_closest_object_of_type_position(controller, target[1], as_tuple=True)
    loc_target = (loc_target[0], loc_target[2])

    true_loc_other = tt.thor_closest_object_of_type_position(controller, other[1], as_tuple=True)
    true_loc_other = (true_loc_other[0], true_loc_other[2])

    heatmap = np.zeros((grid_map.width, grid_map.length))
    for loc_other in grid_map.obstacles:
        score = spcorr.func(loc_target, loc_other, target[0], other[0])
        x_o, y_o = loc_other
        heatmap[x_o, y_o] = score

    plt.imshow(heatmap, cmap="hot")
    plt.scatter([loc_target[0]], [loc_target[1]], marker="o", c="cyan")
    plt.scatter([true_loc_other[0]], [true_loc_other[1]], marker="s", c="green")
    plt.show()


if __name__ == "__main__":
    # _test_simple_correlation_single()
    _test_simple_correlation(scene_type="kitchen")
