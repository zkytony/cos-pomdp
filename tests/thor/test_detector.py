import os
import cv2
import thortils as tt
from thortils.utils.colors import random_unique_color
from thortils.utils.visual import GridMapVisualizer
from cospomdp_apps.thor.detector import GroundtruthDetector, YOLODetector



def _test_groundtruth_detector(scene="FloorPlan1"):
    classes = ["SaltShaker", "Mug", "DishSponge",
               "StoveBurner", "Microwave", "CoffeeMachine", "Fridge", "SoapBottle"]

    controller = tt.launch_controller({'scene':scene})
    grid_map = tt.proper_convert_scene_to_grid_map(controller)
    camera_intrinsic = tt.vision.projection.thor_camera_intrinsic(controller)
    detector = GroundtruthDetector(detectables=classes,
                                   bbox_margin=0.15)
    detections = detector.detect_project(controller.last_event, camera_intrinsic, single_loc=False)
    img = detector.plot_detections(controller.last_event.frame, detections)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    viz = GridMapVisualizer(grid_map=grid_map, obstacle_color=(230, 230, 230))
    _colors = []
    _img = viz.render()
    for d in detections:
        thor_locs = d[-1]
        color = random_unique_color(_colors)
        _img = viz.highlight(_img, thor_locs, color=color, alpha=0.05, thor=True,
                             show_progress=True)  # each cell gets very small alpha
    viz.show_img(_img)
    cv2.imshow("groundtruth", img_bgr)
    cv2.waitKey(0)


def _test_vision_detector(scene="FloorPlan1"):
    from cospomdp_apps.thor.paths import YOLOV5_MODEL_DIR, YOLOV5_DATA_DIR
    scene_type = tt.ithor_scene_type(scene)
    model_path = os.path.join(YOLOV5_MODEL_DIR, f"yolov5-{scene_type}", "best.pt")
    data_config = os.path.join(YOLOV5_DATA_DIR, f"yolov5-{scene_type}", f"yolov5-{scene_type}-dataset.yaml")

    classes = ["SaltShaker", "Mug", "DishSponge",
               "StoveBurner", "Microwave", "CoffeeMachine", "Fridge", "SoapBottle"]

    controller = tt.launch_controller({'scene':scene})
    grid_map = tt.proper_convert_scene_to_grid_map(controller)
    camera_intrinsic = tt.vision.projection.thor_camera_intrinsic(controller)
    detector = YOLODetector(model_path, data_config,
                            detectables=classes,
                            bbox_margin=0.15)
    event = controller.last_event
    camera_pose = tt.thor_camera_pose(event, as_tuple=True)
    detections = detector.detect_project(event.frame, event.depth_frame,
                                         camera_intrinsic, camera_pose)
    img = detector.plot_detections(controller.last_event.frame, detections)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    viz = GridMapVisualizer(grid_map=grid_map, obstacle_color=(230, 230, 230))
    _colors = []
    _img = viz.render()
    for d in detections:
        thor_locs = d[-1]
        color = random_unique_color(_colors)
        _img = viz.highlight(_img, thor_locs, color=color, alpha=0.05, thor=True,
                             show_progress=True)  # each cell gets very small alpha
    viz.show_img(_img)
    cv2.imshow("groundtruth", img_bgr)
    cv2.waitKey(0)

if __name__ == "__main__":
    # _test_groundtruth_detector()
    _test_vision_detector()
