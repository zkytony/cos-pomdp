python -m cospomdp_apps.thor.data.create\
       ../data/yolov5-kitchen\   # path to generated dataset
       --num-train-samples 100\  # number of training samples
       --num-val-samples 30\     # number of validation samples
       --scene-types kitchen     # scene types

python -m cospomdp_apps.thor.data.create\
       ../data/yolov5-living_room\   # path to generated dataset
       --num-train-samples 100\      # number of training samples
       --num-val-samples 30\         # number of validation samples
       --scene-types living_room     # scene types

python -m cospomdp_apps.thor.data.create\
       ../data/yolov5-bedroom\   # path to generated dataset
       --num-train-samples 100\  # number of training samples
       --num-val-samples 30\     # number of validation samples
       --scene-types bedroom     # scene types

python -m cospomdp_apps.thor.data.create\
       ../data/yolov5-bathroom\  # path to generated dataset
       --num-train-samples 100\  # number of training samples
       --num-val-samples 30\     # number of validation samples
       --scene-types bathroom    # scene types
