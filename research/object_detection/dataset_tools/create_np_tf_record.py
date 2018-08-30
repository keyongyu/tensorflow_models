# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import json

from lxml import etree
import PIL.Image
import tensorflow as tf
import numpy
import sys

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw NP dataset.')
flags.DEFINE_boolean('check_bbs', False, 'verify if the boundingbox is out of range')
#flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
#                    'merged set.')
#flags.DEFINE_string('annotations_dir', 'Annotations',
#                    '(Relative) path to annotations directory.')
#flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_empty_images', False, 'will ignore images that has no bbox inside')
FLAGS = flags.FLAGS

def json_to_tf_example(json_data,
                       dataset_directory,
                       label_map_dict ):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    width  = int(json_data.get("image_width"))
    height = int(json_data.get("image_height"))

    filename=orig_filename = json_data.get("filename")
    full_path=orig_full_path = os.path.join(FLAGS.data_dir,"photos", orig_filename)

    with tf.gfile.GFile(orig_full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width,height=image.size

    # filename = json_data.get("filename")+".scaled.jpg"
    # #img_path = os.path.join(FLAGS.data_dir,"photos", filename)
    # full_path = os.path.join(FLAGS.data_dir,"photos", filename)
    # if not os.path.exists(full_path):
    #     #raise ValueError('Please scale image :convert abc.jpg -resize 756x1008 sss.jpg')
    #     orig_filename = json_data.get("filename")
    #     orig_full_path = os.path.join(FLAGS.data_dir,"photos", orig_filename)
    #     #image = PIL.Image.open(orig_full_path)
    #     ##image.resize((756,1008), resample=PIL.Image.BILINEAR).save(full_path)
    #     #image.resize((756,1008), resample=PIL.Image.NEAREST).save(full_path)
    #     os.system("convert "+orig_full_path+" -resize 756x1008 "+full_path)


    #full_path = os.path.join(dataset_directory, img_path)
    #with tf.gfile.GFile(full_path, 'rb') as fid:
    #    encoded_jpg = fid.read()
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width,height=image.size;

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    for obj in json_data.get("bndboxes"):
        if obj.get("id") == "nil":
            continue
        difficult_obj.append(0)
        xmin.append(float(obj.get("x")) / width)
        ymin.append(float(obj.get("y")) / height)
        xmax.append(numpy.clip(float(obj.get("x")+obj.get("w")) / width,0,1))
        ymax.append(numpy.clip(float(obj.get("y")+obj.get("h")) / height,0,1))
        classes_text.append(obj.get("id").encode('utf8'))
        classes.append(label_map_dict[obj.get("id")])
        truncated.append(int(0))
        poses.append("Unspecified")

    width,height=image.size

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example

def check_bndboxes(json_data):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    filename = json_data.get("filename")
    full_path = os.path.join(FLAGS.data_dir,"photos", filename)


    #full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width,height=image.size
    #print ("width="+str(width))
    #width  = int(json_data.get("image_width"))
    #height = int(json_data.get("image_height"))
    for obj in json_data.get("bndboxes"):
        box =obj
        #if type(box["w"]) is str:
        box["w"]=float(box["w"])
        #if type(box["h"]) is str:
        box["h"]=float(box["h"])
        #if type(box["x"]) is str:
        box["x"]=float(box["x"])
        #if type(box["y"]) is str:
        box["y"]=float(box["y"])

        xma=float((obj.get("x")+obj.get("w")) / width)
        yma=float((obj.get("y")+obj.get("h")) / height)
        if xma>1.01 or yma >1.01:
            error_str="Wrong bounding box in "+filename + "(" +str(obj.get("x"))+","+str(obj.get("y"))+","+str(obj.get("w"))+","+str(obj.get("h"))+')'
            error_str = error_str + "image_width="+str(width) + ",image_height=" +str(height)
            print(error_str)
            raise  ValueError(error_str)

def check_bndboxes_dir():
    data_dir = FLAGS.data_dir
    photos_dir= os.path.join(data_dir, "photos")
    annotations_dir = os.path.join(data_dir, "photos","Annotations")
    #examples_list= [f for f in os.listdir(photos_dir) if os.path.isfile(os.path.join(photos_dir, f))]
    examples_list= [f for f in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f))]
    for idx, example in enumerate(examples_list):
        if idx % 50 == 0:
            print('On image %d of %d' %( idx, len(examples_list)))
        #path = os.path.join(annotations_dir, example + '.json')
        path = os.path.join(annotations_dir, example)
        with tf.gfile.GFile(path, 'r') as fid:
            json_str = fid.read()
        json_data=json.loads(json_str)
        tf_example = check_bndboxes(json_data)

    print("no error found")

def to_tfrecord():
    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    photos_dir= os.path.join(data_dir, "photos")
    annotations_dir = os.path.join(data_dir, "photos","Annotations")
    #examples_list= [f for f in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f))]
    examples_list= [f for f in os.listdir(annotations_dir)
                      if not f.startswith(".") and os.path.isfile(os.path.join(annotations_dir, f))
                   ]
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            print('On image %d of %d' %( idx, len(examples_list)))
        #print('load json %s' %(example))
        #sys.stdout.flush()
        #path = os.path.join(annotations_dir, example + '.json')
        path = os.path.join(annotations_dir, example)
        with tf.gfile.GFile(path, 'r') as fid:
            json_str = fid.read()
        json_data=json.loads(json_str)
        #empty=len(json_data.get("bndboxes"))==0

        check_bndboxes(json_data)
        tf_example = json_to_tf_example(json_data, FLAGS.data_dir, label_map_dict )
        writer.write(tf_example.SerializeToString())

    writer.close()

def genereate_label_pbtxt():
    s=""
    dataset_name=""
    with open(os.path.join(FLAGS.data_dir,"templates.json"),"rb") as fp:
        labels= json.load(fp);
        label_def= labels.get('categories', None)[0].get("skus")
        dataset_name=labels.get("datasets")[0].get("name");
        seq=1
        for label in label_def:
            s = s + "item {\n"
            s = s + "  id:" +str(seq)+ "\n"
            s = s + "  name:'" +label.get("id")+ "'\n"
            s = s + "  display_name:'" +label.get("name")+ "'\n"
            s = s + "}\n"
            seq = seq +1


    with open(FLAGS.label_map_path,"wb") as fp:
        fp.write(s)



def main(_):
    if FLAGS.check_bbs:
        check_bndboxes_dir()
    else:
        genereate_label_pbtxt()
        to_tfrecord()
if __name__ == '__main__':
    tf.app.run()
