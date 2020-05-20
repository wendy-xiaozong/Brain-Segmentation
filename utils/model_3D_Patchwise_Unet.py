from tensorflow.keras import Input, regularizers
from tensorflow.keras.layers import Conv3D, AveragePooling3D, BatchNormalization, Conv3DTranspose, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import sys
import os
import numpy as np

from . import batch_norm_3d
from .losses import get_loss

FLAGS = tf.compat.v1.flags.FLAGS


def cnn_3d_segmentation(channels,
                        pool_strides):
    """levels=3,
                              channels=[64, 128, 256],
                              encoder_units=[3, 4, 5],
                              decoder_units=[2, 2],
                              pool_strides=[[2, 2, 2], [1, 2, 2]]"""
    # encoder
    inputs = Input(shape=(8, 24, 24, 3))  # ??
    conv1 = Conv3D(filters=channels[0], kernel_size=3, strides=1, padding="same", dilation_rate=1, activation='relu',
                   kernel_regularizer=regularizers.l2(scale=1.0),
                   bias_regularizer=regularizers.l2(scale=1.0), name='encoder_1')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = AveragePooling3D(pool_size=pool_strides[0], pool_strides=pool_strides[0], padding="same")(conv1)
    conv2 = Conv3D(filters=channels[1], kernel_size=3, strides=1, padding="same", dilation_rate=1, activation='relu',
                   kernel_regularizer=regularizers.l2(scale=1.0),
                   bias_regularizer=regularizers.l2(scale=1.0), name='encoder_2')(conv1)
    conv2 = AveragePooling3D(pool_size=pool_strides[1], pool_strides=pool_strides[1], padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv3D(filters=channels[2], kernel_size=3, strides=1, padding="same", dilation_rate=1, activation='relu',
                   kernel_regularizer=regularizers.l2(scale=1.0),
                   bias_regularizer=regularizers.l2(scale=1.0), name='encoder_3')(conv2)
    conv3 = BatchNormalization()(conv3)
    # decoder
    upconv4 = Conv3DTranspose(filters=channels[2], kernel_size=pool_strides[1], strides=pool_strides[1], padding="same",
                              dilation_rate=1, activation='relu',
                              kernel_regularizer=regularizers.l2(scale=1.0),
                              bias_regularizer=regularizers.l2(scale=1.0), name='decoder_2')(conv3)
    merge4 = Concatenate([conv2, upconv4], axis=4)
    upconv5 = Conv3DTranspose(filters=channels[1], kernel_size=pool_strides[0], strides=pool_strides[0], padding="same",
                              dilation_rate=1, activation='relu',
                              kernel_regularizer=regularizers.l2(scale=1.0),
                              bias_regularizer=regularizers.l2(scale=1.0), name='decoder_1')(merge4)
    merge5 = Concatenate([conv1, upconv5], axis=4)
    conv6 = Conv3D(merge5, filters=11, kernel_size=1, strides=1, padding="same",
                   kernel_regularizer=regularizers.l2(scale=1.0), bias_regularizer=regularizers.l2(scale=1.0),
                   name='FullyConnectedLayer')
    conv6 = tf.nn.softmax(logits=conv6, axis=4, name="softmax")(conv6)  # ???
    model = Model(inputs=inputs, output=conv6)
    return model


def build_model():
    model = cnn_3d_segmentation(channels=[64, 128, 256],
                                pool_strides=[[2, 2, 2], [1, 2, 2]])
    model.summary()

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
