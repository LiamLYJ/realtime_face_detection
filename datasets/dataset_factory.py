from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import wider_face

datasets_map = {
    'wider_face': wider_face,
}

def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        reader)
