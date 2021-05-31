# cos-pomdp

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


## Setting up baselines

### IQA
Run `setup_iqa.sh`.

It will NOT WORK, with the following errors
```
 "nvcc fatal   : Unsupported gpu architecture 'compute_30'"
```
When compiling darknet. Also numerous errors when installing other stuff

Also, getting an error with tensorflow 1.5.0, which has been removed from pypi:
```
"Could not find a version that satisfies the requirement tensorflow-gpu==1.5.0"
```
I used Python 3.8.8, which was allowed by IQA repo's README.

### MJOLNIR
Run `setup_mjolnir.sh`

This one worked. I was able to run their evaluation command,
after downloading `data.zip` and unzip it. The evaluation command is:
```
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/mjolnir_o_pretrain.dat \
    --model MJOLNIR_O \
    --results_json mjolnir_o.json \
    --gpu-ids 0
```
exactly as is w.r.t. their [README](https://github.com/cassieqiuyd/MJOLNIR).

I was also able to get results by running `cat mjolnir_o.json `
```
{
    "GreaterThan/1/spl": 0.2052020242517943,
    "GreaterThan/1/success": 0.637,
    "GreaterThan/5/spl": 0.19212496972596418,
    "GreaterThan/5/success": 0.45535714285714285,
    "done_count": 0.995,
    "ep_length": 19.605,
    "spl": 0.2052020242517943,
    "success": 0.637,
    "total_reward": 3.5747400000000016,
    "total_time": 0.13730125141143798
}
```

However, when running the visualization script, I get
```
Traceback (most recent call last):
  File "visualization.py", line 208, in <module>
    frames = start_controller(args, episode, obj_list, target_parents)
  File "visualization.py", line 148, in start_controller
    img = img_bbx(args, event, ep1[i+4], obj_list, ep1, target_parents)
  File "visualization.py", line 118, in img_bbx
    img2 = cv2.putText(img2, target_parents[2], (20, 200), font, 1, red, 2)
IndexError: list index out of range
```
Also, this code uses AI2-THOR 1.0.1, which is way out of date; The environments look different.

## COS-POMDP

The POMDP itself is clear:

* State: robot state, target object state
* Action: robot movements, environment interactions
* Observation: object attributes
* Transition: ?
* Observation: conditional (relation) + sensor model
* Reward: goal-based


### Actual system

#### Input: RGB/D only

State estimation: RGB/D SLAM
Observation: Object Detector
Transition Model: Learned on top of map
Planning and Control: POMDP solver or Hybrid planning

#### Input: RGB/D + Robot pose

#### Input: RGB/D + Robot pose + Grid map
