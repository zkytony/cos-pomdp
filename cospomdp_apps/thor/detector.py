# Detector using YOLO model
import torch
import numpy as np
import cv2
from PIL import Image
import yaml
from thortils.vision.plotting import plot_one_box
from thortils.vision.general import saveimg
from .common import YOLOV5_REPO_PATH

class Detector:

    def __init__(self, model_path, data_config):
        """
        Args:
            model_path (str): Path to the .pt YOLOv5 model.
                This model should be placed under YOLOV5_REPO_PATH/custom
            data_config (str or dict): loaded from dataset yaml file
                of the dataset used to train the model, or path to
                that yaml file.
        """
        if type(data_config) == str:
            with open(data_config) as f:
                self.config = yaml.load(f, Loader=yaml.Loader)
        else:
            self.config = data_config
        self.classes = self.config["names"]
        self.colors = self.config["colors"]
        self.model_path = model_path
        self.model = torch.hub.load(YOLOV5_REPO_PATH, 'custom',
                                    path=self.model_path,
                                    source="local")

    def detect(self, frame):
        """
        Args:
            frame: RGB image array
        Returns:
            list of (xyxy, conf, cls)
        """
        results = self.model(frame)
        preds = results.pred[0]  # get the prediction tensor
        return [(torch.round(preds[i][:4]).cpu().detach().numpy(),
                 float(preds[i][4]),
                 self.classes[int(preds[i][5])])
                for i in range(len(preds))]

    def save(self, savepath, frame, detections, include=None, conf=True):
        """Given a frame and detections (same format as returned
        by the detect() function), save an image with boxes plotted"""
        img = frame.copy()
        for xyxy, conf, cls in detections:
            if include is not None and cls not in include:
                continue
            if conf:
                label = "{} {:.2f}".format(cls, conf)
            else:
                label = cls
            class_int = self.classes.index(cls)
            plot_one_box(img, xyxy, label, self.colors[class_int],
                         line_thickness=2)
        saveimg(img, savepath)
