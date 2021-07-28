# cos-pomdp

## Organization
```
  cos-pomdp/
     tests/
         # code for tests
     experiments/  # Right now there is nothing here
         # code for experiments
     results
         # experiment / test results
     data/
         # actual data
     external/
         # external repositories (e.g. yolov5, savn, etc.)
     models/
         # actual models
     cosp/
         # core code of COS-POMDP & our algorithm.
         thor/
             code related to thor object search
         vision/
             code related to vision
         utils/
             utilities
```

## YOLOv5 data organization
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

## Create data
```
python -m cosp.vision.data.create data/yolov5 --num-train-samples 3 --num-val-samples 1
```

## Browse generated data
```
python -m cosp.vision.data.browse -m yolo -p path/to/dataset/yaml
```

## Train YOLOv5

## AI2-Thor Setup

Compare with:
- [1] IQA (CVPR'18)
- [2] Visual Semantic Navigation Using Scene Priors (ICRL'19)
- [3] Learning hierarchical relationships for object-goal navigation (CoRL'20)
- [4] Hierarchical and Partially Observable Goal-driven Policy Learning with Goals
  Relational Graph (CVPR'21)


|                  | grid size | h_rotate | v_rotate | fov | object interactions                                  | interaction distance | train/val/test |
|------------------|-----------|----------|----------|-----|------------------------------------------------------|----------------------|----------------|
| [1] IQA          | 0.25      | 90       | 30       | 60  | open, close                                          | 1m                   |                |
| [2] Scene Priors | 0.25      | 45       | 30       | 90  | n/a                                                  | n/a                  | 20/5/5         |
| [3] MJONIR       | 0.25      | 45       | 30       | 100 | n/a                                                  | n/a                  |                |
| [4] HRL-GRG      | 0.25      | 90       | 30       | 90  | n/a                                                  | n/a                  |                |
| [5] ALFRED       | 0.25      | 90       | 15       | 90  | Put, Pickup, Open, Close, ToggleOn, ToggleOff, Slice | 1.5m                 |                |
| ours             | 0.25      | 45       | 30       | 90  | open, close                                          | 1.5m                 |                |


## Dependencies

Check out requirements.txt. This is created under
the `cosp` virtual environment (07-24-2021).
