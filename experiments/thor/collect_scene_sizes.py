# Obtain the size (area and dimension) of each scene.
import os
import thortils as tt
from cospomdp_apps.thor import constants, paths

def load_grid_map(scene, grid_size):
    grid_maps_path = paths.GRID_MAPS_PATH
    gmap_path = os.path.join(grid_maps_path, "{}-{}.json".format(scene, grid_size))
    if os.path.exists(gmap_path):
        print("Loading GridMap from {}".format(gmap_path))
        grid_map = tt.GridMap.load(gmap_path)
    else:
        print(f"Converting scene {scene} to GridMap...")
        controller = tt.launch_controller({**constants.CONFIG, **{"scene": scene}})
        grid_map = tt.proper_convert_scene_to_grid_map(
            controller, grid_size)
        print("Saving grid map to from {}".format(gmap_path))
        os.makedirs(grid_maps_path, exist_ok=True)
        grid_map.save(gmap_path)
    return grid_map


def collect_val_scene_sizes():
    result = {}
    for scene_type in ['kitchen', 'living_room', 'bedroom', 'bathroom']:
        for scene in tt.ithor_scene_names(scene_type, range(21, 31)):
            grid_size = constants.GRID_SIZE
            grid_map = load_grid_map(scene, grid_size)
            dim = (grid_map.width*grid_size, grid_map.length*grid_size)
            free_area = len(grid_map.free_locations) * grid_size**2
            result[scene] = {"dim": dim, "free_area": free_area}
    return result
