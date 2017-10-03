import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

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
    # parser.add_argument("--frozen_model_filename", default="./mob_ssd_net_prepro.pb", type=str, help="Frozen model file to import")
    # parser.add_argument("--frozen_model_filename", default="./optimized_graph.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--frozen_model_filename", default="./tf_files/titan.pb", type=str, help="Frozen model file to import")
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

    # evalure molde for precision and recall
    import re
    from evaluation import evaluate
    file_path = './eval_file_part.txt'
    image_folder = './WIDER_train/images'
    image_path = []
    GT = []
    img_num = 0
    with open(file_path) as f:
        count = 0
        for line in f:
            if (re.search('_',line)) :
                # erase the '\n' in line
                image_path.append(os.path.join(image_folder,line[:-1]))
                count = 0
                img_num += 1
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
    threshold = 0.5

    with tf.Session(graph = graph) as sess:

        for i,j in enumerate(image_path):
            img = cv2.imread(str(image_path[i]))
            height, width,_ = img.shape[:3]
            img = cv2.resize(img,(440,440))
            # destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # input = np.expand_dims(np.array(destRGB),0)
            input = np.expand_dims(np.array(img),0)
            y_out = sess.run(y, feed_dict={
                x: input
            })
            pre = np.ones(y_out.shape)
            pre[:,0] = y_out[:,1] * width
            pre[:,1] = y_out[:,0] * height
            pre[:,2] = y_out[:,3] * width
            pre[:,3] = y_out[:,2] * height
            gt = np.array(GT[i])
            count_precision, count_recall = evaluate(pre,gt,threshold = threshold)
            sum_precision += count_precision
            sum_recall += count_recall
            sum_gt += gt.shape[0]
            sum_pre += pre.shape[0]

    print ('the precision is :', float(sum_precision) / sum_pre)
    print ('the recall is :', float(sum_recall)/ sum_gt)
    print ('totall img:', img_num)

    raise


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
