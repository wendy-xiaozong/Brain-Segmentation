import tensorflow as tf
import sys
import os

import config
from preprocess import batch_norm_3d
from data import get_files, get_objects, add_extra_dims


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
