import os

import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

FACE_LABELS = {
    'none': (0, 'Background'),
    'face': (1, 'Face'),
}

def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions, num_classes):

    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'jpeg'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
