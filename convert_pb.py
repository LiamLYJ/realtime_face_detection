import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import mob_ssd_net
from nets import nets_factory

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

data_format = 'NHWC'
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs_titan/checkpoint'))

with tf.Graph().as_default() as graph:
    input_tensor = tf.placeholder(tf.float32, shape=(None, 440, 440, 3), name='input_image')
    with tf.Session() as sess:
        ssd_net = mob_ssd_net.Mobilenet_SSD_Face()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
            predictions, localisations, _, _ = ssd_net.net(input_tensor, is_training=False)

        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        output_node_names = 'MobilenetV1/Box/softmax_3/Reshape_1,MobilenetV1/Box/softmax_2/Reshape_1,MobilenetV1/Box/softmax_1/Reshape_1,MobilenetV1/Box/softmax/Reshape_1'
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
        with open('./output_graph_nodes.txt', 'w') as f:
            for node in output_graph_def.node:
                f.write(node.name + '\n')

        output_graph = './mob_ssd_net.pb'
        with gfile.FastGFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
