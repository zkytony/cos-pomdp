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
cd cosp/vision
python -m cosp.vision.data.create path/to/output/dataset. -n name
```

# Browse generated data
```
cd cosp/vision
python -m cosp.vision.data.browse -m yolo -p path/to/dataset/yaml
```
