# Create training datasets for YOLOv5 and Faster R-CNN
import os
import yaml
import random
import tqdm
from thortils import *
from .utils import xyxy_to_xywh, saveimg
from ..thor import constants

YOLO_DATA_PATH = "data/yolov5"

def create_yolo_dataset_yaml(datadir):
    content = dict(
        path=datadir,
        train="train",  # training images relative to 'path'
        val="val",      # validation images relative to path
        nc=len(constants.ALL_OBJECT_CLASSES),# number of classes
        names=constants.ALL_OBJECT_CLASSES
    )
    with open(os.path.join(datadir, "dataset.yaml")) as f:
        yaml.dump(content, f)

def generate_yolo_dataset(datadir, scenes, objclasses, for_train,
                          num_samples=100,
                          v_angles=constants.V_ANGLES,
                          h_angles=constants.H_ANGLES):
    for scene in scenes:
        print("Generating YOLO data for", scene)
        generate_yolo_dataset_for_scene(
            datadir, scene, objclasses, for_train,
            num_samples=100,
            v_angles=v_angles,
            h_angles=h_angles)

def generate_yolo_dataset_for_scene(datadir,
                                    scene,
                                    objclasses,
                                    for_train,
                                    num_samples=100,
                                    v_angles=constants.V_ANGLES,
                                    h_angles=constants.H_ANGLES):
    """
    Places the agent at random position within the scene.
    Then, make the agent look around, for all possible
    horizontal and vertical angles. Then grab the frame and object
    bounding boxes within it, until reaching `num_samples`.
    Images with no label for any of the object classes of interest
    will be ignored. The same position will be sampled only once.

    YOLO format: https://towardsdatascience.com/image-data-labelling-and-annotation-everything-you-need-to-know-86ede6c684b1

    The data will be stored in:

        {datadir}/{type}/{scene}-img{#}.jpg
        {datadir}/{type}/{scene}-img{#}.txt

    type is train if `for_train` is True otherwise val.
    scene is the given `scene`. The number # is an integer starting form 0.

    Args:
        for_train (bool): True if use intending to use this dataset for training.
        objclasses (list): List of object classes we want to annotate.
            Note that the order matters. The index of the class in this list
            will be used as the integer class in the YOLO file format.
        v_angles (list): List of acceptable pitch angles
        h_angles (list): List of acceptable yaw angles
    """
    objclasses = {objclasses[i] : i for i in range(len(objclasses))}  # convert the list to dict
    thor_config = {**constants.CONFIG, **{"scene": scene}}
    controller = launch_controller(config)
    reachable_positions = thor_reachable_positions(controller)
    agent_pose = thor_agent_pose(controller.last_event)
    examples = []  # list of (img, annotations)
    y = agent_pose[0]['y']
    roll = agent_pose[1]['z']
    _body_pitch = agent_pose[1]['x']
    _count = 0
    _chosen = set()
    _pbar = tqdm(total=num_sampls)
    while count < num_samples:
        x, z = random.sample(reachable_positions, 1)[0]
        if (x, z) in _chosen:
            continue
        _chosen.add((x,z))
        for pitch in v_angles:
            for yaw in h_angles:
                event = thor_teleport(controller,
                                      position=dict(x=x, y=y, z=z),
                                      rotation=dict(x=_body_pitch, y=yaw, z=roll),
                                      horizon=pitch)  # camera pitch
                img = event.frame
                img_width, img_height = img.shape[:1]
                annotations = []
                for objid in event.instance_detections2D:
                    object_class = thor_object_type(objid)
                    if object_class in objclasses:
                        class_int = objclasses[object_class]
                        bbox2D = event.instance_detections2D[objid]
                        x_center, y_center, w, h =\
                            xyxy_to_xywh(bbox2D, img.shape[:1],
                                         center=True, normalize=True)
                        annotations.append([class_int, x_center, y_center, width_norm, height_norm])
                examples.append((img, annotations))
                _count += 1
                _pbar.update(1)
    # Output the data
    typedir = "train" if for_train else "val"
    os.makedirs(os.path.join(datadir, typedir), exist_ok=True)
    for i, (img, annotations) in enumerate(examples):
        img_path = os.path.join(datadir, typedir, "{}-img{}.jpg".format(scene, i))
        saveimg(img, img_path)
        with open(os.path.join(datadir, typedir, "{}-img{}.txt".format(scene, i))) as f:
            for row in annotations:
                f.write(" ".join(row) + "\n")


if __name__ == "__main__":
    generate_yolo_dataset(YOLO_DATA_PATH,
                          ["FloorPlan1", "FloorPlan2"],
                          constants.KITCHEN_OBJECT_CLASSES,
                          True, num_samples=20)
