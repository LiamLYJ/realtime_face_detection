import tensorflow as tf
from datasets import wider_face_common

slim = tf.contrib.slim

FILE_PATTERN = '%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

SPLITS_TO_SIZES = {
    'test_val': 1,
    'val':2500,
    'train':2000,
}

NUM_CLASSES = 2


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return wider_face_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
