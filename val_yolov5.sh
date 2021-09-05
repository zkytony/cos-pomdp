cd external/
dataset_name="yolov5"
dataset_yaml_file="../data/yolov5/$dataset_name-dataset.yaml"
validation_model_weights="../results/$dataset_name/runs/train/exp2/weights/best.pt"
img_size=600
python yolov5/val.py\
       --data $dataset_yaml_file\
       --img $img_size\
       --weights $validation_model_weights\
       --save-txt\
       --save-hybrid\
       --save-conf\
       --save-json\
       --project "../results/$dataset_name/runs/val"\
       --name "exp"
