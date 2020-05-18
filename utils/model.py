import tensorflow as tf
import sys
import os

from .preprocess import batch_norm_3d

FLAGS = tf.compat.v1.flags.FLAGS


def build_model(inputs, labels):
    x = batch_norm_3d(inputs=inputs, name="input/batch_norm")
    # net = model(x)
    # loss = get_loss(labels=labels,
    #                 predictions=net["output"],
    #                 loss_type=FLAGS.loss_type,
    #                 scope=FLAGS.loss_type,
    #                 huber_delta=FLAGS.huber_delta)
    # dsc = get_dsc(labels=labels,
    #               predictions=net["output"])
    # net["loss"] = loss
    # net["dsc"] = dsc
    # return net
