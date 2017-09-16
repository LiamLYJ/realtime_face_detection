import os
import sys
import random

import numpy as np
import tensorflow as tf
import scipy.io as sio

def read(tfrecords_filename):
    if not isinstance(tfrecords_filename,list):
        tfrecords_filename = [tfrecords_filename]
    filename_queue = tf.train.string_input_producer(tfrecords_filename,num_epochs = 100)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features ={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'label/num_boxes': tf.FixedLenFeature([1], tf.int64),
        }
    )
    ih = tf.cast(features['image/height'],tf.int64)
    iw = tf.cast(features['image/width'],tf.int64)
    num_boxes = tf.cast(features['label/num_boxes'],tf.int64)
    xmin = tf.cast(features['image/object/bbox/xmin'],tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'],tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'],tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'],tf.float32)
    labels = tf.cast(features['image/object/bbox/label'],tf.int64)
    image = tf.decode_raw(features['image/encoded'],tf.uint8)
    image = tf.reshape(image,[ih,iw,3])

    return image,ih,iw,num_boxes,xmin,ymin,xmax,ymax,labels

if __name__ == '__main__':
    with tf.Graph().as_default():
        image,ih,iw,num_boxes,xmin,ymin,xmax,ymax,labels = \
            read('./records/test_val_00000-of-00003.tfrecord')

        image_tmp = tf.cast(image, tf.float32)
        img_4d = tf.expand_dims(image_tmp,0)
        sess = tf.Session()
        init_op = (tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))
        sess.run(init_op)
        tf.train.start_queue_runners(sess = sess)

        image_sum_sample_shape = tf.shape(img_4d)[1:]
        bb = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        tensor_img_with_gt = tf.image.draw_bounding_boxes(img_4d, tf.expand_dims(bb, 0))

        tmp = gt_boxes[:,0]
        with sess.as_default():
            for i in range(2):
                image_, ih_, iw_, ymin_,xmin_,ymax_,xmax_, num_,bb_,results_ = \
                    sess.run([image, ih, iw, ymin,xmin,ymax,xmax, num_boxes,bb,tensor_img_with_gt])
                print ('num:', num_)
                print ('ih: ', ih_)
                print ('iw: ', iw_)
                print ('gt_bb:',bb_)
                import cv2
                cv2.imshow('img',results_[0])
                cv2.waitKey(0)

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
