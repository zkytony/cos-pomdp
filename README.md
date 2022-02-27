# cos-pomdp

COS-POMDP is a POMDP.
This package defines that POMDP and a planner to solve it.
This package also includes instantiation of this POMDP to Ai2Thor for object search.

How should a POMDP project be developed?
You define the POMDP. Then, instantiate it on your domain.
Do all the necessary state (e.g. coordinate) conversions.
You create one or more planner given that POMDP.
Then, you have a POMDP and a solver for it!

Installation: see the Wiki page  ["Setting up COS POMDP project"](https://github.com/zkytony/cos-pomdp/wiki/Setting-up-COS-POMDP-project). This will install `cospomdp` which is the core code for the POMDP components,
and `cospomdp_apps` which contains application to the POMDP to several domains.
After installation is successful, you could test out the basic example:
```
python -m cospomdp_apps.basic.search
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

## AI2-Thor Setup

Compare with:
- [1] IQA (CVPR'18)
- [2] Visual Semantic Navigation Using Scene Priors (ICRL'19)
- [3] Learning hierarchical relationships for object-goal navigation (CoRL'20)
- [4] Hierarchical and Partially Observable Goal-driven Policy Learning with Goals
  Relational Graph (CVPR'21)


|                  | grid size | h_rotate | v_rotate | fov |
|------------------|-----------|----------|----------|-----|
| [1] IQA          | 0.25      | 90       | 30       | 60  |
| [2] Scene Priors | 0.25      | 45       | 30       | 90  |
| [3] MJONIR       | 0.25      | 45       | 30       | 100 |
| [4] HRL-GRG      | 0.25      | 90       | 30       | 90  |
| [5] ALFRED       | 0.25      | 90       | 15       | 90  |
| ours             | 0.25      | 45       | 30       | 90  |

## Dependencies

Check out requirements.txt. This is created under
the `cosp` virtual environment (07-24-2021).


## Citation
```
@inproceedings{zheng2022towards,
  title={Towards Optimal Correlational Object Search,
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  author={Zheng, Kaiyu and Chitnis, Rohan and Sung, Yoonchang and Konidaris, George and Tellex, Stefanie},
  year={2022}
}
```
