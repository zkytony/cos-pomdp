import os
import sys
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
                      ithor_scene_names,
                      ithor_scene_type)
from thortils.constants import KITCHEN_TRAIN_SCENES, KITCHEN_VAL_SCENES
from thortils.vision.metrics import simple_box_iou
from thortils.vision.general import normalized_xywh_to_xyxy
from cospomdp_apps.thor.detector import Detector
from cospomdp_apps.thor.data.browse import yolo_load_info, yolo_load_one
from cospomdp.utils.math import euclidean_dist
from cospomdp.utils.pandas import flatten_index


# Validate vision detector. My own script.
OUTDIR = "../results/true_false_positive_rates"

def run(args):
    # Randomly place the agent in each environment for N times.
    # Then run the detector and record the detections in `results`.
    # Each row is [cls, xyxy, conf, outcome, agent_distance]
    detector = Detector(args.model_path, args.data_yaml)
    results = []

    datadir, fnames, classes, colors = yolo_load_info(args.data_yaml, for_train=False)
    for idx in tqdm(range(len(fnames))):
        img, annots = yolo_load_one(datadir, fnames[idx])
        detections = detector.detect(img)

        # obtain annotated bounding boxes for each frame
        gtbboxes = {}
        for annot in annots:
            class_int = annot[0]
            xywh = annot[1:]
            xyxy = normalized_xywh_to_xyxy(xywh, img.shape[:2], center=True)

            if classes[class_int] not in gtbboxes:
                gtbboxes[classes[class_int]] = []
            gtbboxes[classes[class_int]].append(xyxy)

        # determine TP/FP/TN/FN outcome for the detections of this frame
        for cls in detector.detectable_classes:
            # For each detectable object, if it is not present in gtbboxes,
            # and if it is not present in detections, then we have a True Negative
            # for this class.
            #
            # If it is not present in gtboxes but present in detections, it's FP.
            # If it is present in gtboxes, but not present in detections, it's FN.
            # If it is present in gtboxes, and IOU < thresh, it's FP
            # If it is present in gtboxes, and IOU >= thresh, it's TP
            gtpresent = any(gtcls == cls
                            for gtcls in gtbboxes)
            detpresent = any(d[2] == cls
                             for d in detections)
            if not gtpresent:
                if not detpresent:
                    # true negative
                    results.append([cls, "TN", 1])
                else:
                    # the detected bounding boxes are true positives
                    for xyxy, conf, objtype in detections:
                        if objtype == cls:
                            results.append([cls, "FP", 1])

            else:
                if not detpresent:
                    # false negative
                    results.append([cls, "FN", 1])
                else:
                    # check whether it's a TP and compute detection distance
                    for xyxy, conf, objtype in detections:
                        if objtype == cls:
                            # If the detected bounding box matches any one of the
                            # annotated bounding box for this class, then it is a
                            # true positive.
                            is_tp = False
                            for bbox2D in gtbboxes[cls]:
                                iou = simple_box_iou(bbox2D, xyxy)
                                if iou >= args.iou_thres:
                                    results.append([cls, "TP", 1])
                                    is_tp = True
                                    break
                            if not is_tp:
                                results.append([cls, "FP", 1])

    # Saves results as DataFrame. Use
    df = pd.DataFrame(results,
                      columns=["class", "outcome", "count"])
    df.to_pickle(os.path.join(OUTDIR, "_desired_results_{}.pkl".format(args.scene_type)))
    return detector.detectable_classes

def process(args, detectable_classes):
    df = pd.read_pickle(os.path.join(OUTDIR, "_desired_results_{}.pkl".format(args.scene_type)))
    scounts = df.groupby(["class", "outcome"]).count()
    for index, row in scounts.iterrows():
        cls, outcome = index
        count = row['count']
    rates = []
    for cls in detectable_classes:
        # compute true positive rate and false positive rate
        try:
            tp = scounts.loc[(cls, "TP")]['count']
            fn = scounts.loc[(cls, "FN")]['count']
            fp = scounts.loc[(cls, "FP")]['count']
            tn = scounts.loc[(cls, "TN")]['count']
            true_pos_rate = tp / (tp + fn)
            false_pos_rate = fp / (fp + tn)
            rates.append([cls, true_pos_rate, false_pos_rate])
        except:
            print(f"Cannot get statistic for {cls}")
    dfrates = pd.DataFrame(rates, columns=["class", "TP_rate", "FP_rate"])

    print("##### True Positive and False Positive Counts and Rates ({}) #####".format(args.scene_type))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("--- counts ---")
        print(scounts)
        print("--- rates ---")
        print(dfrates)

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run vision detector in action")
    parser.add_argument("model_path", type=str, help="path to the detector model")
    parser.add_argument("data_yaml", type=str, help="path to the dataset yaml file")
    parser.add_argument("scene_type", type=str, help="scene_type, e.g. kitchen")
    parser.add_argument("--iou-thres", type=float, default=0.7)
    args = parser.parse_args()

    detectable_classes = run(args)
    process(args, detectable_classes)
