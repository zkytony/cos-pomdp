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
from cosp.vision.detector import Detector
from cosp.vision.utils.metrics import simple_box_iou
from cosp.utils.math import euclidean_dist
from cosp.utils.pandas import flatten_index

# Some constant configs
IOU_THRES = 0.5
NUM_SAMPLES_PER_SCENE = 30

# Load detector
MODEL_PATH = "../models/yolov5-25epoch.pt"
DATA_CONFIG = "../data/yolov5/yolov5-dataset.yaml"

def run():
    # Randomly place the agent in each environment for N times.
    # Then run the detector and record the detections in `results`.
    # Each row is [cls, xyxy, conf, outcome, agent_distance]
    detector = Detector(MODEL_PATH, DATA_CONFIG)
    results = []
    for scene in KITCHEN_VAL_SCENES:
        controller = launch_controller(dict(scene=scene))
        for i in tqdm(range(NUM_SAMPLES_PER_SCENE)):
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
                    if thor_object_type(objid) == cls and iou >= IOU_THRES:
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
    df.to_pickle("_desired_results.pkl")

def plot():
    df = pd.read_pickle("_desired_results.pkl")
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.set_ylabel("Distance to Agent (m)")
    order = df.groupby(["class"]).median()["agent_dist"].sort_values().index
    sns.boxplot(x="agent_dist", y="class", data=df,
                orient="h", order=order, palette="light:#5A9")

    # # TP/FP plot
    ax2 = ax.twiny()
    ax2.set_xlabel("Count")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))
    # stripplot is the same as scatterplot except for categorical values.
    sns.stripplot(x="count", y="class", hue="outcome", order=order,
                  data=scounts, ax=ax2, linewidth=1, size=5)
    plt.savefig("_detection_distances.png")

if __name__ == "__main__":
    # run()
    plot()
