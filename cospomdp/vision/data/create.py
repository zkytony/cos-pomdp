# Create training datasets for YOLOv5 and Faster R-CNN
import os
import yaml
import random
from tqdm import tqdm
from thortils import (launch_controller,
                      thor_reachable_positions,
                      thor_agent_pose,
                      thor_teleport,
                      thor_object_type,
                      ithor_scene_names)
from cosp.thor import constants
from ..utils.general import xyxy_to_normalized_xywh, saveimg, make_colors

def yolo_create_dataset_yaml(datadir, classes, name="yolov5"):
    """
    create the yaml file as specified in
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-datasetyaml

    This yaml file will be created at the directory
    where this script is run. The `datadir` should point to the
    directory relative to the current working directory as well.

    Args:
        datadir: Dataset root directory
        classes (list): List of class names
    """
    content = dict(
        path=datadir,
        train="train",    # training images relative to 'path'
        val="val",        # validation images relative to path
        nc=len(classes),  # number of classes
        names=classes,
        colors=make_colors(len(classes), seed=1)
    )
    os.makedirs(datadir, exist_ok=True)
    with open(os.path.join(datadir, "{}-dataset.yaml".format(name)), "w") as f:
        yaml.dump(content, f)

def yolo_generate_dataset(datadir, scenes, objclasses, for_train,
                          num_samples=100,
                          v_angles=constants.V_ANGLES,
                          h_angles=constants.H_ANGLES):
    for scene in scenes:
        print("Generating YOLO data for", scene)
        yolo_generate_dataset_for_scene(
            datadir, scene, objclasses, for_train,
            num_samples=num_samples,
            v_angles=v_angles,
            h_angles=h_angles)

def yolo_generate_dataset_for_scene(datadir,
                                    scene,
                                    objclasses,
                                    for_train,
                                    num_samples=100,
                                    v_angles=constants.V_ANGLES,
                                    h_angles=constants.H_ANGLES):
    """Places the agent at random position within the scene.
    Then, make the agent look around, for all possible
    horizontal and vertical angles. Then grab the frame and object
    bounding boxes within it, until reaching `num_samples`.
    Images with no label for any of the object classes of interest
    will be ignored. The same position will be sampled only once.

    YOLO format: https://towardsdatascience.com/image-data-labelling-and-annotation-everything-you-need-to-know-86ede6c684b1

    The data will be stored in:

        {datadir}/{type}/images/{scene}-img{#}.jpg
        {datadir}/{type}/labels/{scene}-img{#}.txt

    type is train if `for_train` is True otherwise val.  scene is the given
    `scene`. The number # is an integer starting form 0. YOLOv5's documentation
    says: YOLOv5 locates labels automatically for each image by replacing the
    last instance of /images/ in each image path with /labels/

    Args:
        for_train (bool): True if use intending to use this dataset for training.
        objclasses (list): List of object classes we want to annotate.
            Note that the order matters. The index of the class in this list
            will be used as the integer class in the YOLO file format.
        v_angles (list): List of acceptable pitch angles
        h_angles (list): List of acceptable yaw angles

    """
    objclasses = {objclasses[i]: i for i in range(len(objclasses))}  # convert the list to dict
    thor_config = {**constants.CONFIG, **{"scene": scene}}
    controller = launch_controller(thor_config)
    reachable_positions = thor_reachable_positions(controller)
    agent_pose = thor_agent_pose(controller.last_event)
    examples = []  # list of (img, annotations)
    y = agent_pose[0]['y']
    roll = agent_pose[1]['z']
    _body_pitch = agent_pose[1]['x']
    _count = 0
    _chosen = set()
    _pbar = tqdm(total=num_samples)
    while _count < num_samples:
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
                annotations = []
                for objid in event.instance_detections2D:
                    object_class = thor_object_type(objid)
                    if object_class in objclasses:
                        class_int = objclasses[object_class]
                        bbox2D = event.instance_detections2D[objid]
                        x_center, y_center, w, h =\
                            xyxy_to_normalized_xywh(bbox2D, img.shape[:2], center=True)
                        annotations.append([class_int, x_center, y_center, w, h])
                if len(annotations) > 0:
                    examples.append((img, annotations))
                    _count += 1
                    _pbar.update(1)
    _pbar.close()
    controller.stop()
    # Output the data
    partition = "train" if for_train else "val"
    datadir = os.path.join(datadir, partition)
    os.makedirs(os.path.join(datadir, "images"), exist_ok=True)
    os.makedirs(os.path.join(datadir, "labels"), exist_ok=True)
    for i, (img, annotations) in enumerate(examples):
        file_name = "{}-img{}".format(scene, i)
        img_path = os.path.join(datadir, "images", file_name + ".jpg")
        annotations_path = os.path.join(datadir, "labels", file_name + ".txt")
        saveimg(img, img_path)
        with open(annotations_path, "w") as f:
            for row in annotations:
                f.write(" ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    instructions ="""
    python -m cosp.vision.data.create path/to/output/dataset. -n yolov5
    The -n flag supplies name which will be used in {name}-dataset.yaml
    Let's do kitchen first.

    The path/to/output/dataset can be relative or absolute.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Create object detector dataset\n\n" + instructions)
    parser.add_argument("outdir", type=str, help="Path to output directory."
                        "This is the root of the dataset. Will internally use"
                        "the absolute path.")
    parser.add_argument("--model", "-m", type=str, help="Model. Default yolo",
                        default="yolo")
    parser.add_argument("--name", "-n", type=str, help="Name of the dataset."
                        "Default: same name as the model")
    parser.add_argument("--scene-types", type=str, nargs="+", help="Scene types.",
                        default=["kitchen"])
    parser.add_argument("--num-train-samples", type=int, help="Number of training."
                        "samples per scene", default=120)
    parser.add_argument("--num-val-samples", type=int, help="Number of validation."
                        "samples per scene", default=40)
    args = parser.parse_args()
    if args.name is None:
        args.name = args.model

    for scene_type in args.scene_types:
        if scene_type not in constants.SCENE_TYPES:
            raise ValueError("Unrecognized scene type", scene_type)
    object_classes = []  # classes to collect data for
    scenes = {"train": [], "val": []}
    if "kitchen" in args.scene_types:
        object_classes.extend(constants.KITCHEN_OBJECT_CLASSES)
        scenes["train"].extend(constants.KITCHEN_TRAIN_SCENES)
        scenes["val"].extend(constants.KITCHEN_VAL_SCENES)
    if "living_room" in args.scene_types or "living-room" in args.scene_types:
        object_classes.extend(constants.LIVING_ROOM_OBJECT_CLASSES)
        scenes["train"].extend(constants.LIVING_ROOM_TRAIN_SCENES)
        scenes["val"].extend(constants.LIVING_ROOM_VAL_SCENES)
    if "bedroom" in args.scene_types:
        object_classes.extend(constants.BEDROOM_OBJECT_CLASSES)
        scenes["train"].extend(constants.BEDROOM_TRAIN_SCENES)
        scenes["val"].extend(constants.BEDROOM_VAL_SCENES)
    if "bathroom" in args.scene_types:
        object_classes.extend(constants.BATHROOM_OBJECT_CLASSES)
        scenes["train"].extend(constants.BATHROOM_TRAIN_SCENES)
        scenes["val"].extend(constants.BATHROOM_VAL_SCENES)

    print("Building training dataset")
    abs_outdir = os.path.abspath(args.outdir)
    yolo_create_dataset_yaml(abs_outdir, object_classes)
    yolo_generate_dataset(abs_outdir, scenes["train"], object_classes, True,
                          num_samples=args.num_train_samples)
    print("--------------------------------------------------------")
    print("Building val dataset")
    yolo_create_dataset_yaml(abs_outdir, object_classes)
    yolo_generate_dataset(abs_outdir, scenes["val"], object_classes, False,
                          num_samples=args.num_val_samples)
