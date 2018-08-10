if [ "$1" == '' ] || [ "$2" == '' ]  
then
   echo "Usage: $0 np_annotation_dir data_dir"
   exit
fi
ANNO=`realpath "$1"`
mkdir -p "$2/data"
mkdir -p "$2/models"
LABEL_MAP=`realpath "$2/data/label_map.pbtxt"`
TFRECORD=`realpath "$2/data/train.record"`
pushd .
cd ~/Documents/gits/models/research

python object_detection/dataset_tools/create_np_tf_record.py \
    --label_map_path="$LABEL_MAP" \
    --data_dir="$ANNO" \
    --output_path="$TFRECORD"

#python object_detection/dataset_tools/create_pascal_tf_record.py \
#    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
#    --data_dir=/home/keyong/Documents/ssd/VOCdevkit --year=VOC2012 --set=val \
#    --output_path=/home/keyong/Documents/ssd/pascal_val.record

popd
