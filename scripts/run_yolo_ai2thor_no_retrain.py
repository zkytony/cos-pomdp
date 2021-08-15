# Test YOLOv5 on ai2thor images without retrain

import os
import torch
import ai2thor
from tqdm import tqdm
from thortils import launch_controller, thor_place_agent_randomly
from PIL import Image


OUT_DIR = "../results/simple-test-yolov5-ai2thor-no-train"
os.makedirs(OUT_DIR, exist_ok=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
controller = launch_controller({"scene": "FloorPlan1"})
for i in tqdm(range(100)):
    event = thor_place_agent_randomly(controller)
    controller.step(action="Pass")
    results = model(event.frame)
    results.render()
    im = Image.fromarray(results.imgs[0])
    im.save(os.path.join(OUT_DIR, "image-%d.png" % i))
