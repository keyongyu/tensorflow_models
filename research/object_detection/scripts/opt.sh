transform_graph --in_graph=freeze/pb_files/pmi_ukraine/good_annotations/roundy/80126.pb \
--out_graph=freeze/pb_files/pmi_ukraine/good_annotations/roundy/80126_opt.pb \
--inputs='image_tensor' \
--outputs='detection_boxes,detection_scores,detection_classes,num_detections' \
--transforms='
 fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms
 fuse_resize_pad_and_conv
 fuse_pad_and_conv
 fuse_resize_and_conv
  remove_nodes(op=Identity,op=Assert)
  remove_nodes(op=Assert)
  strip_unused_nodes(type=float)
  strip_unused_nodes sort_by_execution_order'

  #round_weights(num_steps=256) 

transform_graph --in_graph=freeze/pb_files/pmi_ukraine/good_annotations/roundy/80126.pb \
--out_graph=freeze/pb_files/pmi_ukraine/good_annotations/roundy/80126_quan.pb \
--inputs='image_tensor' \
--outputs='detection_boxes,detection_scores,detection_classes,num_detections' \
--transforms=' 
  strip_unused_nodes(type=float) 
  remove_nodes(op=CheckNumerics)
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms
  fuse_resize_pad_and_conv
  fuse_pad_and_conv
  fuse_resize_and_conv
  quantize_weights
  quantize_nodes
  strip_unused_nodes
  sort_by_execution_order'

