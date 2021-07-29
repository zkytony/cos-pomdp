import torch
import pandas as pd
import seaborn
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
from cosp.utils.plotting import boxplot_sorted

# Some constant configs
IOU_THRES = 0.5
NUM_SAMPLES_PER_SCENE = 30


# Load detector
MODEL_PATH = "../models/yolov5-25epoch.pt"
DATA_CONFIG = "../data/yolov5/yolov5-dataset.yaml"
detector = Detector(MODEL_PATH, DATA_CONFIG)

# Each row is [cls, xyxy, conf, outcome, agent_distance]
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

df = pd.DataFrame(results,
                  columns=["class", "box", "conf", "outcome", "agent_dist"])
df.to_pickle("desired_results.pkl")

fig, ax = plt.subplots(figsize=(10,6))
fig.tight_layout()
plt.subplots_adjust(bottom=0.3, left=0.2)
boxplot_sorted(df, by=["class"], column="agent_dist", ax=ax)
plt.xticks(rotation=90)
plt.savefig("detection_distances.png")

# print FP and TP counts per class
print(df.groupby(["class", "outcome"]).count()["box"])
