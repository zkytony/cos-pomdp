import os
import argparse
import torch
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from tqdm import tqdm
from thortils import (launch_controller,
                      thor_place_agent_randomly,
                      thor_object_type,
                      thor_agent_position,
                      thor_object_position,
                      ithor_scene_names)
from thortils.constants import KITCHEN_TRAIN_SCENES, KITCHEN_VAL_SCENES
from thortils.vision.metrics import simple_box_iou
from cospomdp_apps.thor.detector import Detector
from cospomdp.utils.math import euclidean_dist
from cospomdp.utils.pandas import flatten_index


OUTDIR = "../results/test_vision_detect_in_action"

def run(args):
    # Randomly place the agent in each environment for N times.
    # Then run the detector and record the detections in `results`.
    # Each row is [cls, xyxy, conf, outcome, agent_distance]
    detector = Detector(args.model_path, args.data_yaml)
    results = []

    scenes = ithor_scene_names(args.scene_type, levels=range(21,31))

    for scene in scenes:
        controller = launch_controller(dict(scene=scene))
        for i in tqdm(range(args.num_samples)):
            event = thor_place_agent_randomly(controller)
            agent_pos = thor_agent_position(event, as_tuple=True)
            detections = detector.detect(event.frame)
            gtbboxes = event.instance_detections2D
            for xyxy, conf, cls in detections:
                # Check if this is correct. Check if any groundtruth
                # bounding box has iou > threshold with xyxy, and if
                # so check if the classes match
                outcome = None
                dist = None
                for objid in gtbboxes:
                    bbox2D = gtbboxes[objid]
                    iou = simple_box_iou(bbox2D, xyxy)
                    if thor_object_type(objid) == cls and iou >= args.iou_thres:
                        # This box corresponds to cls and the detection is close enough
                        # to the groundtruth labeling
                        outcome = "TP" # True positive
                        objpos = thor_object_position(event, objid, as_tuple=True)
                        dist = euclidean_dist(objpos, agent_pos)
                        break
                if outcome is None:
                    # Either cls doesn't appear in the frame but returned box
                    # or cls appears but the detection box doesn;t match. Either way,
                    # it is a false positive
                    outcome = "FP"  # False positive
                results.append([cls, xyxy, conf, outcome, dist])
        controller.stop()

    # Saves results as DataFrame. Use
    df = pd.DataFrame(results,
                      columns=["class", "box", "conf", "outcome", "agent_dist"])
    df.to_pickle(os.path.join(OUTDIR, "_desired_results_{}.pkl".format(args.scene_type)))

def plot(args):
    df = pd.read_pickle(os.path.join(OUTDIR, "_desired_results_{}.pkl".format(args.scene_type)))
    scounts = df.groupby(["class", "outcome"]).count()["box"]
    scounts = flatten_index(scounts).rename(columns={"box": "count"})

    # # Filter rows, based on total count of detections. Only keep ones > 30
    # # Reference: https://stackoverflow.com/a/51589642/2893053
    # def filter_fn1(row):
    #     count = scounts.loc[scounts["class"] == row["class"]].groupby(["class"]).sum().iloc[0]["count"]
    #     return count > 30
    # m = df.apply(filter_fn1, axis=1)
    # df = df[m]
    # classes = set(df["class"].unique())
    # def filter_fn2(row):
    #     return row["class"] in classes
    # m = scounts.apply(filter_fn2, axis=1)
    # scounts = scounts[m]

    fig, ax = plt.subplots(figsize=(12,7))

    fig.tight_layout()
    # adjust spacing https://stackoverflow.com/a/6541454/2893053
    plt.subplots_adjust(bottom=0.125, top=0.9, left=0.17, right=0.95)

    # distance plot
    order = df.groupby(["class"]).median()["agent_dist"].sort_values().index
    sns.boxplot(x="agent_dist", y="class", data=df,
                orient="h", order=order, palette="light:#5A9")
    ax.set_xlabel("Distance to Agent (m)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.grid(axis='y')

    # # TP/FP plot
    ax2 = ax.twiny()
    # stripplot is the same as scatterplot except for categorical values.
    sns.stripplot(x="count", y="class", hue="outcome", order=order,
                  data=scounts, ax=ax2, linewidth=1, size=5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))
    plt.savefig(os.path.join(OUTDIR, "_detection_distances.png"))

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run vision detector in action")
    parser.add_argument("model_path", type=str, help="path to the detector model")
    parser.add_argument("data_yaml", type=str, help="path to the dataset yaml file")
    parser.add_argument("scene_type", type=str, help="scene_type, e.g. kitchen")
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("-n", "--num-samples", type=int, default=30)
    args = parser.parse_args()

    run(args)
    plot(args)
