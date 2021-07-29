# Detector using YOLO model
import torch
import numpy as np
import os
from PIL import Image
import yaml

VISION_MODULE_PATH = os.path.dirname(__file__)
PATH_TO_YOLOV5 = os.path.abspath(os.path.join(VISION_MODULE_PATH, "../../external/yolov5"))

class Detector:

    def __init__(self, model_path, data_config):
        """
        Args:
            model_path (str): Path to the .pt YOLOv5 model.
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
        self.model_path = model_path
        self.model = torch.hub.load(PATH_TO_YOLOV5, 'custom',
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
