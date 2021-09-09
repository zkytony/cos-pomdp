import random
import numpy as np
import thortils as tt
import matplotlib.pyplot as plt
from tqdm import tqdm
from cospomdp.utils.corr_funcs import ConditionalSpatialCorrelation
from cospomdp_apps.thor import constants

def plot_correlation_heatmap(spcorr, target, other, grid_map, controller):
    heatmap = np.zeros((grid_map.length, grid_map.width))
    thor_xo, _, thor_zo = tt.thor_closest_object_of_type_position(controller, other[1], as_tuple=True)
    x_o, z_o = grid_map.to_grid_pos(thor_xo, thor_zo)
    for x_t, z_t in grid_map.obstacles:
        score = spcorr.func((x_t, z_t),
                            (x_o, z_o),
                            target[0], other[0])
        heatmap[z_t, x_t] = score
    plt.imshow(heatmap, cmap="hot")
    plt.scatter(x_o, z_o, marker="o", c="cyan")
    plt.gca().invert_yaxis()
    plt.show()


def _test_simple_correlation_single(scene="FloorPlan1"):
    controller = tt.launch_controller({"scene": scene})
    target = (0, "Bread")
    other = (1, "Bowl")

    distances = tt.thor_distances_in_scene(controller, target[1], other[1])
    print(distances)

    # We want distances in grid cell units
    grid_size = tt.thor_grid_size_from_controller(controller)

    grid_map = tt.proper_convert_scene_to_grid_map(
        controller, grid_size
    )

    distances = np.array(distances) / grid_size
    nearby_thres = constants.NEARBY_THRES / grid_size
    spcorr = ConditionalSpatialCorrelation(target, other, distances, nearby_thres)
    plot_correlation_heatmap(spcorr, target, other, grid_map, controller)


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

    nearby_thres = constants.NEARBY_THRES / grid_size
    spcorr = ConditionalSpatialCorrelation(target, other, all_distances, nearby_thres)

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

    plot_correlation_heatmap(spcorr, target, other, grid_map, controller)

if __name__ == "__main__":
    # _test_simple_correlation_single()
    _test_simple_correlation()
