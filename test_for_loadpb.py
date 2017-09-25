import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--frozen_model_filename", default="./mob_ssd_net.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--frozen_model_filename", default="./mob_ssd_net_prepro.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)

    # for op in graph.get_operations():
    #     print(op.name,op.values())
    # raise
    # note that the names will have a /prefix
    #output_node_names = 'MobilenetV1/Box/softmax_3/Reshape_1,MobilenetV1/Box/softmax_2/Reshape_1,MobilenetV1/Box/softmax_1/Reshape_1,MobilenetV1/Box/softmax/Reshape_1'
    x = graph.get_tensor_by_name('prefix/input_image:0')
    # y = graph.get_tensor_by_name('prefix/MobilenetV1/Box/softmax_3/Reshape_1:0')
    y = graph.get_tensor_by_name('prefix/final_bboxes:0')
    # input = np.ones((1,440,440,3))
    file_path = './test_img/1_Handshaking_Handshaking_1_134.jpg'
    image = Image.open(file_path)
    img = image.resize((440,440), Image.ANTIALIAS)
    input = np.expand_dims(np.array(img),0)
    
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: input
        })
        print(y_out)
    print ("finish, yeah~~")
