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
from cospomdp.utils.math import euclidean_dist, roundany
from .constants import GRID_SIZE
from .paths import YOLOV5_REPO_PATH

class Detector:
    def __init__(self,
                 detectables="any", detection_ranges={},
                 bbox_margin=0.0, visualize=False,
                 detection_sep=GRID_SIZE,
                 max_repeated_detections=100):
        """
        detection_ranges (dict): maps from cls -> expected distance of detection
            detections beyond this range will be dropped.
        """
        self._bbox_margin = bbox_margin
        self.detectable_classes = detectables
        self._visualize = visualize
        self._detection_sep = detection_sep
        self._max_repeated_detections = max_repeated_detections
        self._detection_ranges = detection_ranges
        self._log = {}  # maps from cls -> set(locations it was detected)

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

    def _avg_loc(self, thor_locs):
        return tuple(np.round(np.mean(thor_locs, axis=0), decimals=3))

    def record_detections(self, detections, camera_position, exclude=set()):
        """Given detections, list of (xyxy, conf, cls, thor_locations),
        record the detections' classes & their locations"""
        for d in detections:
            if len(d) == 3:
                print("WARNING: will not record detection because it doesn't contain locations")
                continue
            xyxy, conf, cls, thor_locs = d

            if cls in exclude:
                # do not record this detection (e.g. for target)
                continue

            avg_loc = self._avg_loc(thor_locs)
            if cls not in self._log:
                self._log[cls] = {avg_loc : 1}
            else:
                closest = min(self._log[cls].keys(),
                              key=lambda loc: euclidean_dist(loc, avg_loc))
                if euclidean_dist(closest, avg_loc) <= self._detection_sep:
                    self._log[cls][closest] += 1
                else:
                    self._log[cls][avg_loc] = 1

    def is_overly_repeated_detection(self, detection):
        if len(detection) == 3:
            print("WARNING: cannot check detection repetition it doesn't contain locations")
        xyxy, conf, cls, thor_locs = detection
        avg_loc = self._avg_loc(thor_locs)
        if cls not in self._log:
            return False

        closest = min(self._log[cls].keys(),
                      key=lambda loc: euclidean_dist(loc, avg_loc))
        if euclidean_dist(closest, avg_loc) <= self._detection_sep:
            avg_loc = closest

        if avg_loc not in self._log[cls]:
            return False
        else:
            return self._log[cls][avg_loc] > self._max_repeated_detections

    def within_expected_range(self, detection, camera_position):
        xyxy, conf, cls, thor_locs = detection
        avg_loc = self._avg_loc(thor_locs)
        if cls in self._detection_ranges:
            return euclidean_dist(camera_position, avg_loc) <= self._detection_ranges[cls]
        else:
            print("Expected detection range for {} is unknown. Will include the detection".format(cls))
            return True

    def _accepts(self, detection, camera_position):
        if len(detection) == 3:
            return True
        else:
            return self.within_expected_range(detection, camera_position)\
                and not self.is_overly_repeated_detection(detection)



class YOLODetector(Detector):

    def __init__(self, model_path, data_config,
                 conf_thres=0.4, keep_most_confident=True,
                 **kwargs):
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
        # check the detectable classes are in config
        if not all(c in self.config["names"] for c in self.detectable_classes):
            raise ValueError("Not all detectable objects are handled by the YOLO detector")

        self.classes = self.config["names"]
        self.colors = self.config["colors"]
        self.model_path = model_path
        self._conf_thres = conf_thres
        self._keep_most_confident = keep_most_confident
        print("Loading YOLOv5 vision detector...")
        print(f"    model path: {model_path}")
        print(f"    data config: {data_config}")
        self.model = torch.hub.load(YOLOV5_REPO_PATH, 'custom',
                                    path=self.model_path,
                                    source="local")

    def detect(self, frame):
        """
        Args:
            frame: RGB image array
        """
        results = self.model(frame)
        preds = results.pred[0]  # get the prediction tensor
        detections = [(torch.round(preds[i][:4]).cpu().detach().numpy(),  # bounding box
                       float(preds[i][4]),  # confidence
                       self.classes[int(preds[i][5])])
                      for i in range(len(preds))]
        processed1 = {}
        for d in detections:
            conf = d[1]
            cls = d[2]
            if not self.detectable(cls):
                continue
            if conf >= self._conf_thres:
                if cls not in processed1:
                    processed1[cls] = []
                processed1[cls].append(d)

        processed2 = []
        for cls in processed1:
            if len(processed1[cls]) > 1:
                chosen = max(processed1[cls], key=lambda d: d[1])
                processed2.append(chosen)
            else:
                processed2.append(processed1[cls][0])

        if self._visualize:
            img = self.plot_detections(frame, processed2)
            cv2.imshow("yolov5", img)
            cv2.waitKey(50)
        return processed2

    def detect_project(self, frame, depth_frame, camera_intrinsic, camera_pose):
        bbox_detections = self.detect(frame)
        einv = pj.extrinsic_inv(camera_pose)
        results = []
        for xyxy, conf, cls in bbox_detections:
            xyxy = shrink_bbox(xyxy, self._bbox_margin)
            thor_points = pj.thor_project_bbox(
                xyxy, depth_frame,
                camera_intrinsic, downsample=0.01,
                einv=einv)
            d = (xyxy, conf, cls, thor_points)
            if self._accepts(d, camera_pose[0]):
                results.append(d)
            else:
                print("{} is detected many times at the same location".format(cls))
        return results

class GroundtruthDetector(Detector):
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

        if self._visualize:
            if get_object_ids:
                _viz_detections = [(d[0], d[1], tt.thor_object_type(d[2]))
                                   for d in detections]
            else:
                _viz_detections = detections
            img = self.plot_detections(event.frame, _viz_detections)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("groundtruth", img_bgr)
            cv2.waitKey(50)
        return detections

    def detect_project(self, event, camera_intrinsic=None, single_loc=True):
        bbox_detections = self.detect(event, get_object_ids=True)

        if not single_loc:
            camera_pose = tt.thor_camera_pose(event, as_tuple=True)
            einv = pj.extrinsic_inv(camera_pose)

        results = []
        for xyxy, conf, objectId in bbox_detections:
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
            d = (xyxy, conf, cls, locs)
            if self._accepts(d, camera_pose[0]):
                results.append(d)
            else:
                print("{} is detected many times at the same location".format(cls))
        return results
