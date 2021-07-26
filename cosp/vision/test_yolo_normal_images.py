import os
import torch
from tqdm import tqdm
from PIL import Image


OUT_DIR = "_results/normal"
os.makedirs(OUT_DIR, exist_ok=True)
IMGS_DIR = "_images"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
image_files = os.listdir(IMGS_DIR)
for img_file in tqdm(image_files):
    results = model(os.path.join(IMGS_DIR, img_file))
    results.render()
    im = Image.fromarray(results.imgs[0])
    im.save(os.path.join(OUT_DIR, "image-{}".format(img_file)))
