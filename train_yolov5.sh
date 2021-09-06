cd external/
img_size=600
epocs=30
batch_size=16
pretrained_model_weights="yolov5/yolov5s.pt"
CUDA_VISIBLE_DEVICES=0
GPU=0  # RTX 3080

timestamp() {
  date +"%Y%m%d-%H%M%S" # current time
}
exp_name="exp_$(timestamp)"

train_yolov5()
{
    dataset_name=$1
    dataset_yaml_file="../data/$dataset_name/$dataset_name-dataset.yaml"
    python yolov5/train.py\
           --img $img_size\
           --batch $batch_size\
           --epochs $epocs\
           --data $dataset_yaml_file\
           --weights $pretrained_model_weights\
           --project "../results/$dataset_name/runs/train"\
           --name $exp_name\
           --device $GPU
}

# Train kitchen
train_yolov5 yolov5-kitchen
# train_yolov5 yolov5-living_room
# train_yolov5 yolov5-bedroom
# train_yolov5 yolov5-bathroom
