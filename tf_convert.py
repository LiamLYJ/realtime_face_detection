import tensorflow as tf

from datasets import convert2recoder_wider_face

def main(_):
    convert2recoder_wider_face.run('./','test_val')

if __name__ == '__main__':
    tf.app.run()
