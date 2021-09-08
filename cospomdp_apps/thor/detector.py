# Detector using YOLOv5 model
# Note: to be able to import YOLO, and not get
# "ModuleNotFoundError: No module named 'models.yolo'"
# You may want to check if there is any conflict in you
# sys.path in which there is a module named 'models'.
import torch
import numpy as np
import cv2
from PIL import Image
import yaml
import thortils as tt
from thortils.vision.plotting import plot_one_box
from thortils.vision.general import saveimg, shrink_bbox
from thortils.vision import projection as pj
from thortils.utils.colors import mean_rgb
from .paths import YOLOV5_REPO_PATH

class Detector:
    def __init__(self, detectables="any", bbox_margin=0.15):
        self._bbox_margin = bbox_margin
        self.detectable_classes = detectables

    def detectable(self, cls):
        if self.detectable_classes == "any":
            return True
        else:
            return cls in self.detectable_classes

    def detect(self, inpt):
        """
        Args:
            inpt: input to the detector, e.g. image or event.
        Returns:
            list of (xyxy, conf, cls); We call such a tuple a detection
        """
        raise NotImplementedError

    def detect_project(self, *args, **kwargs):
        """
        Returns:
            list of (xyxy, conf, cls, thor_points), where thor_points is a
            a collection of locations in thor coordinates.
        """
        raise NotImplementedError

    def plot_detections(self, frame, detections, include=None, conf=True):
        """Returns an image with detection boxes plotted;

        `detections` is a list of (xyxy, conf, cls) tuples
        """
        img = frame.copy()
        for detection in detections:
            xyxy, conf, cls = detection[:3]
            if include is not None and cls not in include:
                continue
            if conf:
                label = "{} {:.2f}".format(cls, conf)
            else:
                label = cls
            if hasattr(self, "classes"):
                class_int = self.classes.index(cls)
                color = self.colors[class_int]
            else:
                x1,y1,x2,y2 = xyxy
                color = mean_rgb(frame[y1:y2, x1:x2]).tolist()
            img = plot_one_box(img, xyxy, label, color,
                               line_thickness=2)
        return img

    def save(self, savepath, frame, detections, **kwargs):
        """Given a frame and detections (same format as returned
        by the detect() function), save an image with boxes plotted
        LEGACY."""
        img = self.plot_detections(savepath, frame, detections, **kwargs)
        saveimg(img, savepath)


class YOLODetector(Detector):

    def __init__(self, model_path, data_config, **kwargs):
        """
        Args:
            model_path (str): Path to the .pt YOLOv5 model.
                This model should be placed under YOLOV5_REPO_PATH/custom
            data_config (str or dict): loaded from dataset yaml file
                of the dataset used to train the model, or path to
                that yaml file.
        """
        super().__init__(**kwargs)
        if type(data_config) == str:
            with open(data_config) as f:
                self.config = yaml.load(f, Loader=yaml.Loader)
        else:
            self.config = data_config
        self.classes = self.config["names"]
        self.colors = self.config["colors"]
        self.model_path = model_path
        print("Loading YOLOv5 vision detector...")
        print(f"    model path: {model_path}")
        print(f"    data config: {data_config}")
        self.model = torch.hub.load(YOLOV5_REPO_PATH, 'custom',
                                    path=self.model_path,
                                    source="local")

    @property
    def detectable_classes(self):
        return self.config["names"]

    def detect(self, frame):
        """
        Args:
            frame: RGB image array
        """
        results = self.model(frame)
        preds = results.pred[0]  # get the prediction tensor
        return [(torch.round(preds[i][:4]).cpu().detach().numpy(),
                 float(preds[i][4]),
                 self.classes[int(preds[i][5])])
                for i in range(len(preds))]


class GroundtruthDetector(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, event, get_object_ids=False):
        """
        Args:
            event: ai2thor's Event object.
        Returns
            list of (xyxy, conf, cls); We call such a tuple a detection
        """
        detections = []
        bboxes = tt.vision.thor_object_bboxes(event)  # xyxy
        for objectId in bboxes:
            cls = tt.thor_object_type(objectId)
            if not self.detectable(cls):
                continue
            conf = 1.0
            xyxy = bboxes[objectId]
            if get_object_ids:
                detections.append((xyxy, conf, objectId))
            else:
                detections.append((xyxy, conf, cls))
        return detections

    def detect_project(self, event, camera_intrinsic=None, single_loc=True):
        detections = self.detect(event, get_object_ids=True)

        if not single_loc:
            camera_pose = tt.thor_camera_pose(event, as_tuple=True)
            einv = pj.extrinsic_inv(camera_pose)

        results = []
        for xyxy, conf, objectId in detections:
            cls = tt.thor_object_type(objectId)
            locs = []
            if single_loc:
                # returns only a single location (3D) at the object's position.
                loc3d = tt.thor_object_position(event, objectId, as_tuple=True)
                locs.append(loc3d)
            else:
                # returns grid map cells projected from the bounding box
                xyxy = shrink_bbox(xyxy, self._bbox_margin)
                thor_points = pj.thor_project_bbox(
                    xyxy, event.depth_frame,
                    camera_intrinsic, downsample=0.01,
                    einv=einv)
                locs.extend(thor_points)

            results.append((xyxy, conf, cls, locs))
        return results
