python -m cospomdp_apps.thor.data.create\
       data/yolov5-living_room\
       --num-train-samples 100\
       --num-val-samples 30\
       --scene-types living_room

python -m cospomdp_apps.thor.data.create\
       data/yolov5-kitchen\
       --num-train-samples 100\
       --num-val-samples 30\
       --scene-types kitchen

python -m cospomdp_apps.thor.data.create\
       data/yolov5-bedroom\
       --num-train-samples 100\
       --num-val-samples 30\
       --scene-types bedroom

python -m cospomdp_apps.thor.data.create\
       data/yolov5-bathroom\
       --num-train-samples 100\
       --num-val-samples 30\
       --scene-types bathroom
