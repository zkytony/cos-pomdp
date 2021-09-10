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
    yolov5_training_data_url = "https://drive.google.com/uc?id=1V0GC3wyTsrEAaDfXR0ZyGWLY4j2hYMsx"
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
    yolov5_detector_url = "https://drive.google.com/uc?id=1gfgtLgyLpYa0YLsHkpBF2YWioHR3_ilw"
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
    corr_distances_url = "https://drive.google.com/uc?id=1uKekHoUcIOKPuVfD02YzYTk908SeGKbm"
    gdown.download(corr_distances_url, output, quiet=False)
    cmd=f'''
cd data/thor
unzip {outfname}
cd ../..
'''
    subprocess.check_output(cmd, shell=True)
else:
    print(f"{output} already exists")
