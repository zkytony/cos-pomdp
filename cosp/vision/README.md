# YOLOv5 data organization
```
cosp/vision/data
    yolov5/
        train/
            images/
                FloorPlan10-img0.jpg
                ...
            labels/
                FloorPlan10-img0.txt
                ...
```

# Create data
```
python -m cosp.vision.data.create data/yolov5 --num-train-samples 3 --num-val-samples 1
```

# Browse generated data
```
cd cosp/vision
python -m cosp.vision.data.browse -m yolo -p path/to/dataset/yaml
```
