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
