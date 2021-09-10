"""
Evaluating a detector on a dataset, and compute the true positive,
false positive rates for all classes. It appears that the rates
I computed here, even though over the same validation dataset,
is different from yolov5's confusion matrix plot. The difference
is not too significant. It could be a difference in thresholding,
or someway to treat the input to the network (yolov5 does that in
with multiple pre-processing steps). The iou threshold is 0.7 by
default. I will use my code because I know how it works.

Here is how I count the true/false positive/negatives.

    For each frame,
        For each detectable object class,
            if it is not present in annotated bounding boxes,
                AND if it is not present in detections, then we have a True Negative
                for this class.

                BUT if it is present in detections, all bounding boxes
                    for this class are false positives.

            else (it is present in the annotated bounding box)
                AND if it is present in the detections, then for each
                    detected bounding box, if the IOU >= thresh, it's a TP.
                    otherwise, it is a FP.

                BUT if it is not present in detections, we have a FN.
"""


import os
import sys
import cv2
import argparse
import pandas as pd
from tqdm import tqdm
from thortils.vision.metrics import simple_box_iou
from thortils.vision.general import normalized_xywh_to_xyxy, xyxy_to_normalized_xywh
from cospomdp_apps.thor.detector import YOLODetector
from cospomdp_apps.thor.data.browse import yolo_load_info, yolo_load_one, yolo_plot_one


# Validate vision detector. My own script.
OUTDIR = "../results/true_false_positive_rates"

def run(args):
    # Randomly place the agent in each environment for N times.
    # Then run the detector and record the detections in `results`.
    # Each row is [cls, xyxy, conf, outcome, agent_distance]
    detector = YOLODetector(args.model_path, args.data_yaml,
                            conf_thres=0.0, keep_most_confident=False)
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
                            results.append([cls, "FP_off", 1])

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
                            _max_iou = float('-inf')
                            for bbox2D in gtbboxes[cls]:
                                iou = simple_box_iou(bbox2D, xyxy)
                                _max_iou = max(_max_iou, iou)
                                if iou >= args.iou_thres:
                                    results.append([cls, "TP", 1])
                                    is_tp = True
                                    break
                            if not is_tp:
                                ######## DEBUGGING
                                if args.debug and cls == "StoveBurner":
                                    xywh = xyxy_to_normalized_xywh(xyxy, img.shape[:2])
                                    annots = []
                                    for bbox2D in gtbboxes[cls]:
                                        xywh_gt = xyxy_to_normalized_xywh(bbox2D, img.shape[:2])
                                        annots.append([classes.index(cls), *xywh_gt])
                                    img = yolo_plot_one(img,
                                                        annots,
                                                        classes,
                                                        colors)
                                    old_color = colors[classes.index(cls)]
                                    colors[classes.index(cls)] = [200.0, 0.0, 0.0]
                                    img = yolo_plot_one(img,
                                                        [[classes.index(cls), *xywh]],
                                                        classes,
                                                        colors)
                                    colors[classes.index(cls)] = old_color
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    cv2.imshow(cls, img_rgb)
                                    cv2.waitKey(0)

                                results.append([cls, "FP", 1])
                                if _max_iou <= 0.05:
                                    results.append([cls, "FP_off", 1])

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
            fp_off = scounts.loc[(cls, "FP_off")]['count']
            true_pos_rate = tp / (tp + fn)
            false_pos_rate = fp / (fp + tn)
            rates.append([cls, tp, fn, true_pos_rate, fp, fp_off, tn, false_pos_rate])
        except:
            print(f"Cannot get statistic for {cls}")
    dfrates = pd.DataFrame(rates, columns=["class", "TP", "FN", "TP_rate", "FP", "FP_off", "TN", "FP_rate"])

    print("##### True Positive and False Positive Counts and Rates ({}) #####".format(args.scene_type))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("--- counts ---")
        print(scounts.sort_values(dfrates.columns[0]))
        print("--- rates ---")
        dfrates = dfrates.sort_values(dfrates.columns[0])
        print("|{}|".format("|".join(dfrates.columns)))
        for index, row in dfrates.iterrows():
            print("|{}|".format("|".join(map(str, row))))


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run vision detector in action")
    parser.add_argument("model_path", type=str, help="path to the detector model")
    parser.add_argument("data_yaml", type=str, help="path to the dataset yaml file")
    parser.add_argument("scene_type", type=str, help="scene_type, e.g. kitchen")
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    detectable_classes = run(args)
    process(args, detectable_classes)
