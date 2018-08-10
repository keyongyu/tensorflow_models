if [ "$1" == '' ] || [ "$2" == '' ]  
then
   echo "Usage: $0 np_annotation_dir splitted_dir"
   exit
fi
SRC_DIR=`realpath "$1"`
TO_DIR=`realpath "$2"`

SRC_TEMPLATE_JSON="$SRC_DIR/templates.json"
TO_TEMPLATE_JSON="$TO_DIR/templates.json"

pushd .
cd ~/Documents/gits/models/research

python object_detection/dataset_tools/split_big_image.py \
    --src_dir="$SRC_DIR" \
    --to_dir="$TO_DIR"
cp "$SRC_TEMPLATE_JSON" "$TO_TEMPLATE_JSON"
#python object_detection/dataset_tools/create_pascal_tf_record.py \
#    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
#    --data_dir=/home/keyong/Documents/ssd/VOCdevkit --year=VOC2012 --set=val \
#    --output_path=/home/keyong/Documents/ssd/pascal_val.record

popd
