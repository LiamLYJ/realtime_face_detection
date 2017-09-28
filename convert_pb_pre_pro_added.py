import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image
import cv2
from nets import mob_ssd_net
from nets import nets_factory
from nets import tf_methods
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from preprocessing import owndata_preprocessing


def draw_results(img, rclasses, rscores, rbboxes):
    img = np.array(img)
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
    cv2.imwrite('./pb_convert_img/test.jpg', img)

data_format = 'NHWC'
net_shape = (440, 440)
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/checkpoint'))
select_threshold = 0.06
nms_threshold = 0.25

with tf.Graph().as_default() as graph:
    input_tensor = tf.placeholder(tf.float32, shape=(None, 440, 440, 3), name='input_image')
    # input_for_preprocess = tf.squeeze(input_tensor)
    # only chooses the first frame
    input_for_preprocess = input_tensor[0]
    input_for_preprocess = tf.cast(input_for_preprocess, tf.uint8)
    image_pre, labels_pre, bboxes_pre, bbox_img = owndata_preprocessing.preprocess_for_eval(
        input_for_preprocess, None, None, net_shape, data_format, resize=owndata_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    with tf.Session() as sess:
        ssd_net = mob_ssd_net.Mobilenet_SSD_Face()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)
            # print ('len of predictions :', len(predictions))
            # print ('predictions shape:', predictions[0].shape)
            ssd_anchors = ssd_net.anchors(net_shape)

            _classes, _scores, _bboxes = tf_methods.tf_ssd_bboxes_select(
                    predictions, localisations, ssd_anchors,
                    select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)


            bboxes = tf_methods.tf_bboxes_clip(bbox_img, _bboxes)

            # print ('shape pf classes:',classes.shape)
            # print ('shape of scores:',scores.shape)
            # print ('shape of bboxes:',bboxes.shape)
            # print ('**************')
            # raise

            classes = tf.squeeze(_classes)
            scores = tf.squeeze(_scores)
            # I just comment this !!!!!!!!!!!!!!!
            # some bugs in sort methods need to be solved !!!!!!
            # classes, scores, bboxes = tf_methods.tf_bboxes_sort(classes, scores, bboxes, top_k=400)
            classes, scores, bboxes = tf_methods.tf_bboxes_nms(classes, scores, bboxes, nms_threshold=nms_threshold)
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            bboxes = tf_methods.tf_bboxes_resize(bbox_img, bboxes)

            scores_ = tf.identity(scores,'final_scores')
            bboxes_ = tf.identity(bboxes,'final_bboxes')
            classes_ = tf.ones_like(scores_)

        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


        # try feed forward graph
        file_path = './train_img/20_Family_Group_Family_Group_20_33.jpg'
        image = Image.open(file_path)
        img = image.resize((440,440), Image.ANTIALIAS)
        input = np.expand_dims(np.array(img),0)
        # run threshold selected nodes
        # rclasses, rscores, rbboxes,rpredictions, rlocalisations = sess.run(
        #     [_classes,_scores,_bboxes,predictions, localisations], feed_dict={
        #     input_tensor: input
        # })
        # draw_results(img,rclasses,rscores,rbboxes)
        # print ('shape of rpredictions:',rpredictions[0].shape)
        # print ('shape of rlocalisations:',rlocalisations[0].shape)
        # print (rclasses)
        # print ('shape of rclasses:',rclasses.shape)
        # print (rscores)
        # print ('shape of rscores:',rscores.shape)
        # print (rbboxes)
        # print ('shape of rbboxes:',rbboxes.shape)
        # run after nms nodes
        rclasses_,rscores_, rbboxes_ = sess.run(
            [classes_,scores_,bboxes_], feed_dict={
            input_tensor: input
        })
        print('rscores_:',rscores_)
        print('rbboxes:',rbboxes_)
        draw_results(image,rclasses_,rscores_,rbboxes_)
        raise

        output_node_names = 'final_bboxes,final_scores'
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))

        with open('./output_graph_nodes_prepro.txt', 'w') as f:
            for node in output_graph_def.node:
                f.write(node.name + '\n')

        output_graph = './mob_ssd_net_prepro.pb'
        with gfile.FastGFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
