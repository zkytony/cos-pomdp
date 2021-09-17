import random
import os
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pprint import pprint
from cospomdp.utils.corr_funcs import ConditionalSpatialCorrelation
from cospomdp_apps.thor import constants
import thortils as tt

DISTS_DIR = "../data/thor/corr_dists"

def plot_correlation_heatmap(spcorr, target_class, other_class,
                             grid_map, target_loc="random", rnd=random):
    heatmap = np.zeros((grid_map.length, grid_map.width))

    if target_loc == "random"
        x_target, y_target = rnd.sample(grid_map.obstacles, 1)[0]
    else:
        x_target, y_target = target_loc

    for x_i, y_i in grid_map.obstacles:
        score = spcorr.func((x_target, y_target),
                            (x_i, z_i),
                            target_class, other_class)
        heatmap[x_i, y_i] = score
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.scatter(x_o, z_o, marker="o", c="cyan")
    plt.gca().invert_yaxis()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Get a plot of correlational model")
    parser.add_argument("scene_type", type=str, description="scene type, e.g. kitchen")
    parser.add_argument("target_type", type=str, description="target type")
    parser.add_argument("corr_type", type=str, description="correct, learned, wrong")
    args = parser.parse_args()

    # Loads the database
    corr_objects = {}
    for fname in os.listdir(DISTS_DIR):
        if fname.startswith("corr-dist"):
            tokens = fname.split("_")
            scene_type = tokens[1]
            target, corrobj = tokens[2].split("-")
            scene = tokens[3]
            corr_type = tokens[4].split(".pkl")[0]

            if corr_type == args.corr_type\
               and target == args.target_type:
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

        pprint(list(corr_objects[corrobj].keys()))
        scene = input("Which scene? ")
        while scene not in corr_objects[corrobj]:
            scene = input(f"{scene} is not in the list. Try again: ")

        fname = corr_objects[corrobj][scene]
        with open(os.path.join(DISTS_DIR, fname), 'rb') as f:
            distances = pickle.load(fname)

        reverse = args.corr_type == "wrong"
        learned = args.corr_type == "learned"
        nearby_thres = constants.NEARBY_THRES / constants.GRID_SIZE
        spcorr = ConditionalSpatialCorrelation(target, corrobj, distances,
                                               nearby_thres, reverse=reverse,
                                               learned=learned)
        print("{}: Average distance between {} and {} is {:.3f}"\
              .format(args.scene_type, target, corrobj, spcorr._mean_dist))

        controller = tt.launch_controller({**constants.CONFIG, **{"scene":scene}})
        grid_map = tt.proper_convert_scene_to_grid_map(
            controller, constants.GRID_SIZE
        )
        plot_correlation_heatmap(spcorr, target, corrobj,
                                 grid_map, target_loc="random",
                                 rnd=random.Random(100))

if __name__ == "__main__":
    main()
