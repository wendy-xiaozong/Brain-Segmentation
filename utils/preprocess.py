import tensorflow as tf
import numpy as np


# ??
def batch_norm_3d(inputs):
    batch = tf.transpose(inputs, perm=[1, 2, 3, 0, 4])  # before [None, 8, 24, 24, 3] after [None, 8, 24, 24, 3]
    mean, var = tf.nn.moments(batch, axes=[0, 1, 2])
    batch = tf.nn.batch_normalization(batch,
                                      mean=mean,
                                      variance=var,
                                      offset=0,
                                      scale=1,
                                      variance_epsilon=1e-9)
    batch = tf.transpose(batch, perm=[3, 0, 1, 2, 4])  # after [None, 8, 24, 24, 3]
    return batch
