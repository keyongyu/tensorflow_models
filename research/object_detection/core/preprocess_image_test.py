import tensorflow as tf
import matplotlib.pyplot as plt
import functools
import json
import os
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from object_detection.utils import config_util
from object_detection import inputs
from object_detection.builders import dataset_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import model_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import ops as util_ops
from object_detection.utils import shape_utils

tf.enable_eager_execution()

img=plt.imread("/home/keyong/Documents/dataset/RB_Total/train_RB_flag/photos/001.JPG")

tf_img=tf.convert_to_tensor(img)

tf_img=tf.image.flip_left_right(tf_img)




#imgplot = plt.imshow(tf_img)
#plt.show()


def main(_):
    configs = config_util.get_configs_from_pipeline_file(
        '/home/keyong/Documents/ssd/tf_records/RB_Total/models/ssd_mobilenet_v2.config')


    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']


    #def get_next(config):
    #    return dataset_builder.make_initializable_iterator(
    #        dataset_builder.build(config)).get_next()

    #dataset= dataset_builder.build(input_config)

    #iterator = dataset.make_one_shot_iterator()
    #input_dict = iterator.get_next()

    train_input_fn=inputs.create_train_input_fn(train_config, input_config, model_config)

    dataset=train_input_fn({})

    iterator = dataset.make_one_shot_iterator()
    img = None
    for input_tuple in iterator:
        input_dict=input_tuple[0]  #hash, image, true_image_shape
        input_dict2=input_tuple[1]  #groundtruth_area (1, 100),
                                    # groundtruth_boxes[1, 100, 4]
                                    #groundtruth_classes[1,100,51]
                                    #groundtruth_is_crowd... [1,100]
                                    #groundtruth_weights... [1,100]
                                    #num_groundtruth_boxes (1,)
        normalized_img= tf.squeeze(input_dict["image"],axis=[0]) #value varies from [-1.0, 1.0]

        img=(normalized_img+1.0) /2
        #img=normalized_img


        break
    ffff=10;

    imgplot = plt.imshow(img)
    plt.show()





if __name__ == '__main__':
    tf.app.run()
