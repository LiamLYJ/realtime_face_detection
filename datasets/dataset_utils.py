from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def to_tfexample_raw(image_data, height, width,gt_boxes,num_boxes ):
    # gt_boxes shape of (x1,y1, w,h,5 )
    boxes = gt_boxes
    boxes[:,2] = gt_boxes[:,0] + gt_boxes[:,2]
    boxes[:,3] = gt_boxes[:,1] + gt_boxes[:,3]
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in boxes:
        assert len(b) == 4
        [l.append(point) for l,point in zip([xmin, ymin, xmax, ymax],b)]
    labels = [1]*num_boxes
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/shape': int64_feature([height,width,3]),
     #   'label/gt_boxes': bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2,1)
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'label/num_boxes':int64_feature(num_boxes), # how many boxes in this image
    }))
