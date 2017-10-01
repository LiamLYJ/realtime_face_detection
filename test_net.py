import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET

import sys
sys.path.append('./')

from nets import ssd_common, np_methods
from nets import mob_ssd_net
from preprocessing import owndata_preprocessing

# path for testing data
# RESULT_PATH = './result_train'
RESULT_PATH = './result_test'

LABELS = ["None", "Face"]

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
# use gpu options
# gpu_options = tf.GPUOptions(allow_growth=True)
# config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# use cpu options
cpu_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)
# sess = tf.InteractiveSession(config=config)
sess = tf.InteractiveSession(config = cpu_conf)

# Input placeholder.
net_shape = (440, 440)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = owndata_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=owndata_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'MobilenetV1' in locals() else None
ssd_net = mob_ssd_net.Mobilenet_SSD_Face()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.image("Image", image_4d))
    f_i = 0
    # print ('sec one predict_map shape :', predictions[1].shape)
    for predict_map in predictions:
        # print ('shape of predict_map:', predict_map.shape)
        # raise
        predict_map = predict_map[:, :, :, :, 1:]
        predict_map = tf.reduce_max(predict_map, axis=4)
        # if f_i < 3:
        if f_i < 4:
            predict_list = tf.split(predict_map, 7, axis=3)
            anchor_index = 1
            for anchor in predict_list:
                summaries.add(tf.summary.image("predicte_map_%d_anchor%d" % (f_i,anchor_index), tf.cast(anchor, tf.float32)))
                anchor_index += 1
        # else:
        #     predict_map = tf.reduce_max(predict_map, axis=3)
        #     predict_map = tf.expand_dims(predict_map, -1)
        #     summaries.add(tf.summary.image("predicte_map_%d" % f_i, tf.cast(predict_map, tf.float32)))
        f_i += 1
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


# Restore SSD model.
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/checkpoint'))
# if that checkpoint exists, restore from checkpoint
saver = tf.train.Saver()
summer_writer = tf.summary.FileWriter("./logs_test/", sess.graph)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.06, nms_threshold=0.25, net_shape=(440, 440)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img, summary_op_str = sess.run([image_4d, predictions, localisations, bbox_img, summary_op],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    # print ('shape pf classes:',rclasses.shape)
    # print (rclasses)
    # print ('shape of scores:',rscores.shape)
    # print (rscores)
    # print ('shape of bboxes:',rbboxes.shape)
    # print (rbboxes)
    # print ('**************')
    # raise
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    # print ('2_rclasses:' ,rclasses)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    # print ('3_rclasses:' ,rclasses)
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    # summer_writer.add_summary(summary_op_str, 1)
    # print rclasses, rscores, rbboxes
    return rclasses, rscores, rbboxes

def draw_results(img, rclasses, rscores, rbboxes, index, img_name):

    height, width, channels = img.shape[:3]
    for i in range(len(rclasses)):
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
        # cv2.putText(img, LABELS[rclasses[i]], (xmin, ymin),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
        cv2.putText(img, str(rscores[i]), (xmin, ymin),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)

        cv2.imwrite('./%s/test_%d.jpg' %(result_path,index), img)


# evalure molde for precision and recall
import re
from evaluation import evaluate
file_path = './eval_file.txt'
image_folder = './WIDER_train/images'
image_path = []
GT = []

with open(file_path) as f:
    count = 0
    for line in f:
        if (re.search('_',line)) :
            # erase the '\n' in line
            image_path.append(os.path.join(image_folder,line[:-1]))
            count = 0
        elif len(line.split(' ')) < 2 :
            num = int(line)
            boxes = np.ones((num,4))
        else :
            tmp = line.split(' ')
            boxes[count,0] = int(tmp[0])
            boxes[count,1] = int(tmp[1])
            boxes[count,2] = int(tmp[0]) + int(tmp[2])
            boxes[count,3] = int(tmp[1]) + int(tmp[3])
            count += 1
            if count == num :
                GT.append(boxes)

sum_pre = 0
sum_gt = 0
sum_precision = 0
sum_recall = 0
threshold = 0.1

for i,j in enumerate(image_path):
    img = cv2.imread(str(image_path[i]))
    height, width,_ = img.shape[:3]
    destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rclasses, rscores, rbboxes =  process_image(destRGB)
    pre = np.ones(rbboxes.shape)
    pre[:,0] = rbboxes[:,1] * width
    pre[:,1] = rbboxes[:,0] * height
    pre[:,2] = rbboxes[:,3] * width
    pre[:,3] = rbboxes[:,2] * height
    gt = np.array(GT[i])
    count_precision, count_recall = evaluate(pre,gt,threshold = threshold)
    sum_precision += count_precision
    sum_recall += count_recall
    sum_gt += gt.shape[0]
    sum_pre += pre.shape[0]

print ('the precision is :', float(sum_precision) / sum_pre)
print ('the recall is :', float(sum_recall)/ sum_gt)

raise

# draw testing results 
result_path = RESULT_PATH
if not os.path.exists(result_path):
    os.mkdir(result_path)

if result_path == './result_train':
    img_path = './train_img/'
else:
    img_path = './test_img/'

image_names = []
for root, dirs, files in os.walk(img_path):
    for file in files:
        image_names.append(os.path.join(root,file))
# print image_names

index = 1
for image_name in image_names[1:]:
    img = cv2.imread(image_name)
    destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rclasses, rscores, rbboxes =  process_image(destRGB)
    # print ("rclasses:", rclasses)
    draw_results(img, rclasses, rscores, rbboxes, index, image_name)
    index += 1
