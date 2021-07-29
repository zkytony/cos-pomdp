import os
import torch
from tqdm import tqdm
from PIL import Image


OUT_DIR = "../results/simple-test-yolov5-normal"
os.makedirs(OUT_DIR, exist_ok=True)

image_files = ["https://i.imgur.com/9Bi9zR9.jpeg",
                "https://i.imgur.com/yjX5wkX.jpeg",
                "https://i.imgur.com/5nF5qjH.jpeg",
                "https://i.imgur.com/FqhJyHO.jpeg",
                "https://i.imgur.com/5sqm3PM.jpeg",
                "https://i.imgur.com/IyC9C8m.jpeg"]
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
for i, img_file in enumerate(tqdm(image_files)):
    results = model(img_file)
    results.render()
    im = Image.fromarray(results.imgs[0])
    im.save(os.path.join(OUT_DIR, f'image{i}.jpeg'))
