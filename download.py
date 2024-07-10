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

import os
import gdown
import subprocess
import yaml
import pathlib
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

outfname = "yolov5-training-data.zip"
output = f"data/{outfname}"
if not os.path.exists(output):
    print("Downloading yolov5 training data")
    yolov5_training_data_url = "https://drive.google.com/uc?id=1hAdoD4-mcCbVovR4aO2qASN4EXbWsx8p"
    gdown.download(yolov5_training_data_url, output, quiet=False)
    cmd=f'''
cd data/
unzip {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")

# Go into each dataset file, update the yaml file's path
for name in os.listdir("data"):
    if name.startswith("yolo"):
        config_file = f"data/{name}/{name}-dataset.yaml"
        if not os.path.exists(config_file):
            continue
        with open(config_file) as f:
            config = yaml.load(f)
        if config is not None:
            config['path'] = os.path.join(ABS_PATH, "data", name)
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            print(f"Updated path in yaml for {name}")

outfname = "yolov5-detectors.zip"
output = f"models/{outfname}"
if not os.path.exists(output):
    print("Downloading yolov5 detectors")
    yolov5_detector_url = "https://drive.google.com/uc?id=1gOdgkOeLnLB0v4PAeA4pJ9IslfpOP6Ih"
    gdown.download(yolov5_detector_url, output, quiet=False)
    cmd=f'''
cd models/
unzip {outfname}
cd ..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")

# Create symbolic link to the best model
for name in os.listdir("models"):
    if os.path.isdir(f"models/{name}"):
        exp_name = os.listdir(f"models/{name}")[0]

        src = os.path.abspath(f"models/{name}/{exp_name}/weights/best.pt")
        dst = os.path.abspath(f"models/{name}/best.pt")
        if not os.path.exists(dst):
            os.symlink(src, dst)
        print(f"Linked models for {name}")

os.makedirs("data/thor", exist_ok=True)
outfname = "corrs.zip"
output = f"data/thor/{outfname}"
if not os.path.exists(output):
    print("Downloading corr distances")
    corr_distances_url = "https://drive.google.com/uc?id=1btyqud0KP1pvuUpGr7_h8PB0t_Yfkyzo"
    gdown.download(corr_distances_url, output, quiet=False)
    cmd=f'''
cd data/thor
unzip {outfname}
cd ../..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
