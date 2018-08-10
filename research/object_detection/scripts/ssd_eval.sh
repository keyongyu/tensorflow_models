if [ "$1" == '' ] || [ "$2" == "" ]
then
   echo "Usage: $0 data_dir ssd_mbv2"
   echo "Usage: $0 data_dir frcnn"
   exit
fi

if [ "$2" == "ssd_mbv2" ]
then
    PIPELINE=`realpath "$1/models/ssd_mobilenet_v2.config"`
else
    PIPELINE=`realpath "$1/models/faster_rcnn_inception_v2.config"`
fi

TRAIN_DIR=`realpath "$1/train_dir"`
#PIPELINE=`realpath "$1/models/ssd_mobilenet_v1.config"`
EVAL_DIR=`realpath "$1/eval_dir"`

#python ~/Documents/gits/models/research/object_detection/train.py \
#    --logtostderr \
#    --pipeline_config_path="$PIPELINE" \
#    --train_dir="$TRAIN_DIR"

export CUDA_VISIBLE_DEVICES='2'
python ~/Documents/gits/models/research/object_detection/legacy/eval.py \
    --logtostderr \
    --pipeline_config_path="$PIPELINE" \
    --checkpoint_dir="$TRAIN_DIR" \
    --eval_dir="$EVAL_DIR"
