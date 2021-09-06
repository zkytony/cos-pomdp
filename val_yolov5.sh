cd external/
dataset_name="yolov5-kitchen"
run="exp_20210905-160954"
dataset_yaml_file="../data/$dataset_name/$dataset_name-dataset.yaml"
validation_model_weights="../results/$dataset_name/runs/train/$run/weights/last.pt"
img_size=600
CUDA_VISIBLE_DEVICES=0
GPU=0  # RTX 3080
python yolov5/val.py\
       --data $dataset_yaml_file\
       --img $img_size\
       --weights $validation_model_weights\
       --save-txt\
       --save-hybrid\
       --save-conf\
       --save-json\
       --project "../results/$dataset_name/runs/val"\
       --name "$run"\
       --iou-thres 0.7\
       --device $GPU
