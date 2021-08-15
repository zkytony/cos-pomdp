import os
from tqdm import tqdm
from thortils import (launch_controller,
                      thor_place_agent_randomly)
from thortils.constants import KITCHEN_TRAIN_SCENES, KITCHEN_VAL_SCENES
from cosp.vision.detector import Detector

# Some constant configs
IOU_THRES = 0.5
NUM_SAMPLES_PER_SCENE = 30

# Load detector
MODEL_PATH = "../models/yolov5-25epoch.pt"
DATA_CONFIG = "../data/yolov5/yolov5-dataset.yaml"

OUTDIR = "../results/test_detector_quick"
SMALL_OBJECTS = [
    "Apple",
    "Potato",
    "Tomato",
    "Egg",
    "Lettuce",
    "PepperShaker",
    "ButterKnife",
    "Cup",
    "Fork",
    "Ladle",
    "SaltShaker",
    "Spatula",
]
MEDIUM_OBJECTS = [
    "Pot",
    "Kettle",
    "Bread",
    "Book",
    "Bowl",
    "Toaster",
    "Statue",
    "Pan",
    "CoffeeMachine",
    "Microwave",
]
LARGE_OBJECTS = [
    "Chair",
    "Cabinet",
    "Drawer",
    "DiningTable",
    "CoffeeTable",
    "Fridge",
    "Shelf",
    "CounterTop",
    "StoveBurner",
    "Sink"
]

CLASSES = LARGE_OBJECTS
TYPE = "val"
NUM = 2
def run():
    detector = Detector(MODEL_PATH, DATA_CONFIG)
    scene = eval(f"KITCHEN_{TYPE.upper()}_SCENES")[NUM]
    controller = launch_controller(dict(scene=scene))
    for i in tqdm(range(10)):
        event = thor_place_agent_randomly(controller)
        detections = detector.detect(event.frame)
        detector.save(os.path.join(OUTDIR, "image-%s-%d-%d.jpg" % (TYPE, NUM, i)),
                      event.frame, detections, include=CLASSES)

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run()
