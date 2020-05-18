import tensorflow as tf
import numpy as np


def batch_norm_3d(inputs, name=None):
    with tf.name_scope(name):
        print("before input:", inputs.shape.as_list())
        batch = tf.transpose(inputs, perm=[1, 2, 3, 0, 4])
        print("after input:", inputs.shape.as_list())
        mean, var = tf.nn.moments(batch, axes=[0, 1, 2])
        batch = tf.nn.batch_normalization(batch,
                                          mean=mean,
                                          variance=var,
                                          offset=0,
                                          scale=1,
                                          variance_epsilon=1e-9)
        batch = tf.transpose(batch, perm=[3, 0, 1, 2, 4])
        print("after batch:", batch.shape.as_list())
    return batch
