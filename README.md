# cos-pomdp

COS-POMDP is a POMDP.
This package defines that POMDP and a planner to solve it.
This package also includes instantiation of this POMDP to Ai2Thor for object search.

How should a POMDP project be developed?
You define the POMDP. Then, instantiate it on your domain.
Do all the necessary state (e.g. coordinate) conversions.
You create one or more planner given that POMDP.
Then, you have a POMDP and a solver for it!

Installation
```
pip install -e .
```
This will install `cospomdp` which is the core code for the POMDP components,
and `cospomdp_apps` which contains application to the POMDP to several domains.

Test basic example:
```
python -m cospomdp_apps.basic.example
```

### Caveats
The external methods, e.g. SAVN, MJOLNIR, are placed under `cos-pomdp/external`.
However, for importability, a symbolic link to the `cos-pomdp/external/mjolnir`
directory is created under `cos-pomdp/cospomdp_apps/thor/mjolnir`. Please
make sure that link points to the right absolute path on your computer.
For example, you can directly create a new one by:
```
cd repo/cos-pomdp/cospomdp_apps/thor
ln -sf $HOME/repo/cos-pomdp/external/mjolnir/ mjolnir
```

## Organization
Contains two packages: `cospomdp` and `cospomdp_apps`.
The former defines the COS-POMDP (domain, models, agent, planner)
and the latter applies COS-POMDP to specific applications
such as ai2thor.

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
python -m cospomdp_apps.thor.data.create data/yolov5 --num-train-samples 3 --num-val-samples 1
```

## Browse generated data
```
python -m cospomdp_apps.thor.data.browse -m yolo path/to/dataset/yaml
```
Use `a` and `d` to browse back and forth. To quit, press `q`.

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
