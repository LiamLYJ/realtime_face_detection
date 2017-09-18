import os
import sys
import random
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from datasets import dataset_factory
slim = tf.contrib.slim

batch_size = 10
num_readers = 4
# dataset = dataset_factory.get_dataset(
#     'wider_face', 'test_val', './records')
dataset = dataset_factory.get_dataset(
    'wider_face', 'train', './records')
with tf.Graph().as_default():
    sess = tf.InteractiveSession()
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers = num_readers,
        common_queue_capacity = 20 * batch_size,
        common_queue_min = 10 * batch_size,
        shuffle = True
    )
    [image, shape, glabels, gbboxes,num] = provider.get(['image', 'shape', 'object/label', 'object/bbox','num'])

    image = tf.cast(image, tf.float32)
    img_4d = tf.expand_dims(image,0)
    w = tf.cast(shape,tf.float32)[1]
    h = tf.cast(shape,tf.float32)[0]
    # if already convert to [0,1]
    w=1.0
    h=1.0

    y1 = gbboxes[:,0] / h
    y2 = gbboxes[:,2] / h
    x1 = gbboxes[:,1] / w
    x2 = gbboxes[:,3] / w
    tensor_img_with_gt = tf.image.draw_bounding_boxes(img_4d,
                                tf.expand_dims(tf.stack((y1,x1,y2,x2),axis = 1), 0))
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):

        print ('shape; glabel,gboxes,num_box:',sess.run([shape,glabels,gbboxes,num]))
        # import matplotlib.pyplot as plt
        # tensor_img_with_gt = tf.cast(tensor_img_with_gt,tf.uint8)
        # plt.imshow(tensor_img_with_gt.eval()[0])
        # plt.show()
raise

data = sio.loadmat('/Users/liuyongjie/Desktop/realtime_face_detection/wider_face_split/wider_face_val.mat')
# tmp0 = data['event_list'][0][0][0]
# tmp1 = data['event_list'][1][0][0]
# print (tmp0)
# print (tmp1)
# _file_list = data['file_list'][1][0]
# print (_file_list.shape)
# _file = _file_list[0][0][0]
# _file1 = _file_list[1][0][0]
# print (_file)
# print (_file1)
# _face_box_list = data['face_bbx_list'][0][0]
# face = _face_box_list[0][0][0]
# print (face)
# face1 = _face_box_list[0][0][1]
# print (face1)
# face_all = _face_box_list[0][0]
# print ('************')
# print (face_all)
# print (_face_box_list.size)
# face_bbox = _face_box_list[0][0]
# print (face_bbox.shape)
# tmp_bbox = face_bbox[0,:]
# print (tmp_bbox)
# _blur_label_list = data['blur_label_list'][0][0]
# print (_blur_label_list.size)
# blur_label = _blur_label_list[0][0]
# print (blur_label.shape)
# tmp_blur = blur_label[1,:]
# print (tmp_blur)
pose_label_list = data['pose_label_list']
pose_label = pose_label_list[0][0][0][0]
print (pose_label)
