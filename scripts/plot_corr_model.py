import random
import os
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
from pprint import pprint
from cospomdp.utils.corr_funcs import ConditionalSpatialCorrelation
from cospomdp_apps.thor import constants
import thortils as tt

DISTS_DIR = "../data/thor/corrs"

def plot_correlation_heatmap(spcorr, target_class, other_class,
                             grid_map, target_loc="random", rnd=random):
    heatmap = np.zeros((grid_map.length, grid_map.width))

    if target_loc == "random":
        x_target, y_target = rnd.sample(grid_map.obstacles, 1)[0]
    else:
        x_target, y_target = target_loc

    for x_i, y_i in grid_map.obstacles:
        score = spcorr.func((x_target, y_target),
                            (x_i, y_i),
                            target_class, other_class)
        heatmap[y_i, x_i] = score
        print(score)

    plt.imshow(heatmap, cmap="Greys_r")
    plt.colorbar()
    plt.scatter(x_target, y_target, marker="o", c="cyan")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Get a plot of correlational model")
    parser.add_argument("scene_type", type=str, help="scene type, e.g. kitchen")
    parser.add_argument("target_type", type=str, help="target type")
    parser.add_argument("corr_type", type=str, help="correct, learned, wrong")
    parser.add_argument("-f", "--scene", "--floor-plan", type=str, help="scene /FloorPlan to test on")
    parser.add_argument("-t", "--target", type=str, help="Target class")
    args = parser.parse_args()

    if args.scene_type.startswith("living"):
        args.scene_type = "living+room"

    # Loads the database
    corr_objects = {}
    for fname in os.listdir(DISTS_DIR):
        fname = fname.replace("living_room", "living+room")
        if fname.startswith("distances"):
            tokens = fname.split("_")
            scene_type = tokens[1]
            target, corrobj = tokens[2].split("-")
            scene = tokens[3].split(".json")[0]
            if target == args.target_type:
                if corrobj not in corr_objects:
                    corr_objects[corrobj] = {}
                corr_objects[corrobj][scene] = fname

    if len(corr_objects) == 0:
        print(f"Looks like the data for {args.target_type} does not exist.")

    else:
        pprint(list(corr_objects.keys()))
        corrobj = input("Which correlated object class? ")
        while corrobj not in corr_objects:
            corrobj = input(f"{corrobj} is not in the list. Try again: ")

        if args.scene is None:
            pprint(list(corr_objects[corrobj].keys()))
            scene = input("Which scene? ")
            while scene not in corr_objects[corrobj]:
                scene = input(f"{scene} is not in the list. Try again: ")
        else:
            scene = args.scene

        fname = corr_objects[corrobj][scene]
        if args.corr_type == "learned":
            fname = corr_objects[corrobj]["train"]

        with open(os.path.join(DISTS_DIR,
                               fname.replace("living+room", "living_room")),
                               'rb') as f:
            dd = json.load(f)
            if args.corr_type == "learned":
                dd = dd[0]
            distances = np.asarray(dd["distances"]) / constants.GRID_SIZE

        reverse = args.corr_type == "wrong"
        learned = args.corr_type == "learned"
        nearby_thres = constants.NEARBY_THRES / constants.GRID_SIZE
        spcorr = ConditionalSpatialCorrelation(args.target_type, corrobj, distances,
                                               nearby_thres, reverse=reverse,
                                               learned=learned)
        print("{}: Average distance between {} and {} is {:.3f}"\
              .format(args.scene_type, args.target_type, corrobj, spcorr._mean_dist))

        controller = tt.launch_controller({**constants.CONFIG, **{"scene":scene}})
        if scene.startswith("Floor"):
            grid_map = tt.proper_convert_scene_to_grid_map(
                controller, constants.GRID_SIZE
            )
            plot_correlation_heatmap(spcorr, args.target_type, corrobj,
                                     grid_map, target_loc="random",
                                     rnd=random.Random(100))
        else:
            print("Only doing validation scenes, for now.")

if __name__ == "__main__":
    main()
