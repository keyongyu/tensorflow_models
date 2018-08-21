import tensorflow as tf
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
from object_detection.utils import config_util
import matplotlib.pyplot as plt
from object_detection import inputs

tf.enable_eager_execution()
#
#img=plt.imread("/home/keyong/Documents/dataset/RB_Total/train_RB_flag/photos/001.JPG")
#
#tf_img=tf.convert_to_tensor(img)
#
##tf_img=tf.image.flip_left_right(tf_img)
#
#tf_img2=tf.image.adjust_contrast(tf_img, 2)
#tf_img3=tf.image.adjust_saturation(tf_img, 2)
#
# fig, axs = plt.subplots(3, 1)
# axs[0].imshow(tf_img)
# axs[1].imshow(tf_img2)
# axs[2].imshow(tf_img3)
# plt.show()
#tf_img3=tf.image.adjust_contrast(tf_img, 1.25)

# #imgplot = plt.imshow(tf_img)
#
# imgplot2 = plt.imshow(tf_img2)
# #imgplot = plt.imshow(tf_img3)
# plt.show()

def mainx(_):
    aaa=tf.convert_to_tensor([1,3])
    bbb=tf.stack([1,3])
    #bbb=tf.expand_dims([1,3],[1])
    #print(aaa)
    #print(bbb)
    y_hat = tf.convert_to_tensor(np.array([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]]))
    y_true = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    y_hat_softmax = tf.nn.softmax(y_hat)
    print("0.....",y_hat_softmax )
    print("x.....",- y_true * tf.log(y_hat_softmax) )
    #sparse_loss_ce=tf.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_hat)
    #print("1.5.....", sparse_loss_ce)
    loss_per_instance_1 = -tf.reduce_sum(y_true * tf.log(y_hat_softmax), reduction_indices=[1])
    print("1.....", loss_per_instance_1)


    loss_per_instance_2 = -tf.reduce_sum(y_true * tf.nn.log_softmax(y_hat), reduction_indices=[1])
    print("2....", loss_per_instance_2)

    loss_per_instance_2 = - y_true * tf.nn.log_softmax(y_hat)
    print("2....", loss_per_instance_2)

    sss=tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true)
    print("3....", sss)
    #sess.run(y_hat)

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
    imgs =[]
    idx=0
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
        imgs.append(img)
        #img=normalized_img
        idx+=1
        if idx >8:
            break

    ffff=10;
    fig, axs = plt.subplots(len(imgs)/3,3)
    for idx , img in enumerate(imgs):
       axs[idx/3,idx%3].imshow(img)
    plt.show()





if __name__ == '__main__':
    tf.app.run()
