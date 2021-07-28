cd external/
DATASET_NAME="yolov5"
DATASET_YAML_FILE="../data/yolov5/$DATASET_NAME-dataset.yaml"
VALIDATION_MODEL_WEIGHTS="../results/$DATASET_NAME/runs/train/weights/best.pt"
IMG_SIZE=600
python yolov5/val.py\
       --data $DATASET_YAML_FILE\
       --img $IMG_SIZE\
       --weights $VALIDATION_MODEL_WEIGHTS\
       --save-txt\
       --save-hybrid\
       --save-conf\
       --save-json\
       --project "../results/$DATASET_NAME/runs/val"\
       --name "exp"
