import os
import sys
import random
import math
import numpy as np
import tensorflow as tf
import scipy.io as sio
from PIL import Image

'''
about wider_face_dataset
blur:
  clear->0
  normal blur->1
  heavy blur->2

expression:
  typical expression->0
  exaggerate expression->1

illumination:
  normal illumination->0
  extreme illumination->1

occlusion:
  no occlusion->0
  partial occlusion->1
  heavy occlusion->2

pose:
  typical pose->0
  atypical pose->1

invalid:
  false->0(valid image)
  true->1(invalid image)

The format of txt ground truth.
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
'''

from datasets.dataset_utils import \
    int64_feature, float_feature, bytes_feature, to_tfexample_raw

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'wider_face_split'
# DIRECTORY_IMAGES = 'WIDER_val/images'
DIRECTORY_IMAGES = 'WIDER_train/images'
# ANNOTATION_NAME = 'wider_face_val.mat'
ANNOTATION_NAME = 'wider_face_train.mat'
# TFRecords convertion parameters.
NUM_SHARDS = 5000

def get_num_images(file_list):
    counter = 0
    counter_list = []
    for i in range(file_list.shape[0]):
        counter = counter + file_list[i][0].shape[0]
        # counter_list.append(file_list[i][0].shape[0])
        counter_list.append(counter)
    return counter,counter_list


def get_index_from_id(id,num_list):
    tmp_num_list = num_list[:]
    if id < tmp_num_list[0]:
        index = [0,id]
        return index
    if id in tmp_num_list:
        index = [tmp_num_list.index(id) + 1, 0]
    else:
        tmp_num_list.append(id)
        tmp_num_list.sort()
        index = [tmp_num_list.index(id) ,id - num_list[tmp_num_list.index(id) -1]]
    return index


def filter_with_other_label(face_bbx,blur_label,expression_label,illumination_label,
                                   invalid_label,occlusion_label,pose_label):
    dump_blur = blur_label == 2
    dump_expression = expression_label == 1
    dump_illumination = illumination_label == 1
    dump_invalid = invalid_label == 1
    dump_occlusion = occlusion_label ==2
    dump_pose = pose_label == 1
    # any of w or h < 16 will be dump
    dump_scale = np.logical_or(face_bbx[:,2]<20, face_bbx[:,3]<20)

    dump = dump_scale
    # over_occluded is not considered
    dump = np.logical_or(dump,dump_occlusion)
    # over_blured is not considerd
    dump = np.logical_or(dump,dump_blur)
    # bad expression is not considered
    dump = np.logical_or(dump,dump_expression)
    # over_illuminated is not considered
    # dump = np.logical_or(dump,dump_illumination)
    # invalid is not considered
    # dump = np.logical_or(dump,dump_invalid)
    # atypical is not considered
    dump = np.logical_or(dump,dump_pose)
    # too small box is not considered
    dump = np.logical_or(dump,dump_scale)
    keep = np.where(dump == False)[0]
    gt_boxes = face_bbx[keep,:]
    # print ('after fileted:', gt_boxes)
    return gt_boxes


def add_to_tfrecord(record_dir, dataset_dir, annotation_path, dataset_split_name):
    mat_file = sio.loadmat(annotation_path)
    file_list = mat_file['file_list']
    blur_label_list = mat_file['blur_label_list']
    event_list = mat_file['event_list']
    expression_label_list = mat_file['expression_label_list']
    face_bbx_list = mat_file['face_bbx_list']
    illumination_label_list = mat_file['illumination_label_list']
    invalid_label_list = mat_file['invalid_label_list']
    occlusion_label_list = mat_file['occlusion_label_list']
    pose_label_list = mat_file['pose_label_list']

    num,num_list = get_num_images(file_list)

    cats = 2
    print ('%s has %d images' % (dataset_split_name,num))

    num_shards = int(num/float(NUM_SHARDS))
    num_per_shard = int(math.ceil(num/float(num_shards)))
    counter  =0
    for shard_id in range(num_shards):
        record_filename = get_output_filename(record_dir,
                        dataset_split_name,shard_id,num_shards)
        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, num)
            for image_id in range(start_ndx, end_ndx):
                if image_id % 10 == 0:
                    sys.stdout.write('\r>> Converting image %d/%d shard %d\n' %(image_id+1,num,shard_id))
                    sys.stdout.flush()
                index = get_index_from_id(image_id,num_list)
                # print ('********************')
                # print ('curent index is:', index)
                # print ('current id is :', image_id)
                # print (num_list)
                filename = file_list[index[0]][0][index[1]][0][0]
                face_bbx = face_bbx_list[index[0]][0][index[1]][0]

                event = event_list[index[0]][0][0]
                blur_label = blur_label_list[index[0]][0][index[1]][0]
                expression_label = expression_label_list[index[0]][0][index[1]][0]
                illumination_label = illumination_label_list[index[0]][0][index[1]][0]
                invalid_label = invalid_label_list[index[0]][0][index[1]][0]
                occlusion_label = occlusion_label_list[index[0]][0][index[1]][0]
                pose_label = pose_label_list[index[0]][0][index[1]][0]

                file_path = os.path.join(dataset_dir,DIRECTORY_IMAGES,event,filename+'.jpg')
                # print (file_path)
                img_raw = tf.gfile.FastGFile(file_path,'r').read()
                # img = np.array(Image.open(file_path))
                # img_raw = img.tostring()

                img_shape = np.array(Image.open(file_path)).shape
                height = img_shape[0]
                width = img_shape[1]
                # fileer gt_boxes with some extra label from gt in dataset
                gt_boxes = filter_with_other_label(face_bbx,blur_label,expression_label,illumination_label,
                                                   invalid_label,occlusion_label,pose_label)
                num_boxes = gt_boxes.shape[0]
                if num_boxes == 0:
                    print('no valid bbox here, dumped')
                    continue
                # gt_boxes_raw = gt_boxes.tostring()
                # print (gt_boxes)
                # print (gt_boxes.shape)
                # raise

                # put image_data, height, width,gt_boxes,num_boxes into tfrecorder
                try:
                    # example = to_tfexample_raw(img_raw,height,width,gt_boxes_raw,num_boxes)
                    example = to_tfexample_raw(img_raw,height,width,gt_boxes,num_boxes)
                except:
                    print ('\n***********dumped')
                    continue
                tfrecord_writer.write(example.SerializeToString())
                counter = counter + 1
            tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print ('Counter = %d'%counter)

def get_output_filename(record_dir, dataset_split_name, shard_id,num_shards):
    output_filename = '%s_%05d-of-%05d.tfrecord' % \
            (dataset_split_name, shard_id, num_shards)
    return os.path.join(record_dir,output_filename)

def run(dataset_dir, dataset_split_name, shuffling=False):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    record_dir = os.path.join(dataset_dir,'records')
    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)
    annotation_path = os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS,ANNOTATION_NAME)
    add_to_tfrecord(record_dir,dataset_dir,annotation_path,dataset_split_name)
    print('\nFinished converting the wider face dataset!')
