#!/usr/bin/env bash

if [ "$1" == '' ] || [ "$2" == '' ]  || [ "$3" == '' ]  
then
   echo "Usage: $0 train_dir seq_no dest_pb_file"
   exit
fi

#TRAIN_DIR="/home/keyong/Documents/ssd/train_dir"
#SEQ_NO="120341"
TRAIN_DIR=$1
SEQ_NO=$2
OPT_PB_FILE=$3
echo python ~/Documents/gits/models/research/object_detection/export_inference_graph.py --input_type \
image_tensor --pipeline_config_path "$TRAIN_DIR/pipeline.config"  --trained_checkpoint_prefix \
"$TRAIN_DIR/model.ckpt-$SEQ_NO" --output_dir /tmp/my.pb

python ~/Documents/gits/models/research/object_detection/export_inference_graph.py \
--pipeline_config_path "$TRAIN_DIR/pipeline.config" --trained_checkpoint_prefix    \
"$TRAIN_DIR/model.ckpt-$SEQ_NO" --input_type image_tensor --output_directory /tmp/my.pb  \

transform_graph --in_graph=/tmp/my.pb/frozen_inference_graph.pb  \
    --out_graph="$OPT_PB_FILE" \
    --inputs='image_tensor' \
    --outputs='detection_boxes,detection_scores,detection_classes,num_detections' \
    --transforms='
      fold_constants(ignore_errors=true) 
      fold_batch_norms 
      fold_old_batch_norms 
      strip_unused_nodes'

#remove_nodes(op=Identity,op=DecodeProtoV2,op=EncodeProto)
#transform_graph --in_graph=/tmp/my.pb/frozen_inference_graph.pb  \
#    --out_graph="$OPT_PB_FILE" \
#    --inputs='image_tensor' \
#    --outputs='detection_boxes,detection_scores,detection_classes,num_detections' \
#    --transforms='
#  add_default_attributes
#  strip_unused_nodes(type=float)
#  remove_nodes(op=CheckNumerics)
#  fold_constants(ignore_errors=true)
#  fold_batch_norms
#  fold_old_batch_norms
#  fuse_resize_pad_and_conv
#  fuse_pad_and_conv
#  fuse_resize_and_conv
#  quantize_weights
#  quantize_nodes
#  strip_unused_nodes'
#  #sort_by_execution_order'

rm -fr /tmp/my.pb
