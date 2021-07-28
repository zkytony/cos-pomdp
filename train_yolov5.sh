cd external/
DATASET_NAME="yolov5"
DATASET_YAML_FILE="../data/yolov5/$DATASET_NAME-dataset.yaml"
IMG_SIZE=600
EPOCS=20
BATCH_SIZE=16
PRETRAINED_MODEL_WEIGHTS="yolov5/yolov5s.pt"

python yolov5/train.py\
       --img $IMG_SIZE\
       --batch $BATCH_SIZE\
       --epochs $EPOCS\
       --data $DATASET_YAML_FILE\
       --weights $PRETRAINED_MODEL_WEIGHTS\
       --project "../results/$DATASET_NAME/runs/train"\
       --name "exp"
