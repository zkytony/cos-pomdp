# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
