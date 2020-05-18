import tensorflow as tf
import sys
import os
import numpy as np

from . import batch_norm_3d

FLAGS = tf.compat.v1.flags.FLAGS


def encoder(levels,
            channels,
            number_of_units,
            pool_strides,
            name):
    """levels=3,
                      channels=[64, 128, 256],
                      number_of_units=[3, 4, 5],
                      pool_strides=pool_strides"""
    model = tf.keras.Input(shape=(1, 8, 24, 24, 3))  # ??
    for level in range(levels):
        model = get_layer(
            model,
            number_of_units=number_of_units[level],
            channels=channels[level],
            name="")
        if level < (levels - 1):
            model = tf.keras.layers.AveragePooling3D(
                pool_size=pool_strides[level],
                strides=pool_strides[level],
                padding="same")(model)
        return model


def transition_layer(net,
                     input_scope,
                     scope,
                     levels,
                     channels):
    for level in range(levels - 1):
        net[scope + "_level_" + str(level)] = get_layer( \
            net=net[input_scope + "_level_" + str(level)],
            name="level_" + str(level),
            number_of_units=1,
            channels=channels[level])
    net[scope + "_level_" + str(levels - 1)] = \
        net[input_scope + "_level_" + str(levels - 1)]
    return net


def cnn_3d_segmentation(levels,
                        channels,
                        encoder_units,
                        decoder_units,
                        pool_strides):
    """levels=3,
                              channels=[64, 128, 256],
                              encoder_units=[3, 4, 5],
                              decoder_units=[2, 2],
                              pool_strides=[[2, 2, 2], [1, 2, 2]]"""
    transition_channels = list((np.array(channels) * 0.25).astype(np.int32))
    net = encoder(levels=levels,
                  channels=channels,
                  number_of_units=encoder_units,
                  pool_strides=pool_strides)
    net = transition_layer(net=net,
                           input_scope="encoder",
                           scope="transition",
                           levels=levels,
                           channels=transition_channels)
    net = decoder(net=net,
                  input_scope="transition",
                  scope="decoder",
                  levels=levels,
                  channels=channels,
                  number_of_units=decoder_units,
                  type_of_layer="cnn",
                  pool_strides=pool_strides)
    net = output_layer(net=net,
                       input_scope="decoder",
                       scope="output")
    return net


def get_layer(model,
              number_of_units,
              channels,
              name):
    """type_of_layer=cnn,
                number_of_units=3, // for 3 kind of picture
                channels=64"""
    for n in range(number_of_units):
        model = tf.keras.layers.Conv3D(filters=channels,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",  # same代表卷积时如果原始数据边缘不足的卷积核大小的，就自动填0补齐。
                                       dilation_rate=1,
                                       activation=tf.nn.relu,
                                       kernel_regularizer=tf.keras.regularizers.l2(scale=1.0),
                                       bias_regularizer=tf.keras.regularizers.l2(scale=1.0))(model)
    model = batch_norm_3d(model)
    for n in range(number_of_units):
        tmp = model
        for _ in range(2):
            net = tf.keras.layers.Conv3D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         dilation_rate=1,
                                         activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2(scale=1.0),
                                         bias_regularizer=tf.keras.regularizers.l2(scale=1.0))(model)
            net = batch_norm_3d(net)  # ???
        net = batch_norm_3d(net)
    return net


def cnn_3d_segmentation_1():
    net = cnn_3d_segmentation(levels=3,
                              channels=[64, 128, 256],
                              encoder_units=[3, 4, 5],
                              decoder_units=[2, 2],
                              pool_strides=[[2, 2, 2], [1, 2, 2]])
    return net


def model():
    # if FLAGS.model == "resnet_3d_1":
    #     net = resnet_3d_segmentation_1(inputs=inputs)
    # else:
    net = cnn_3d_segmentation_1(inputs=inputs)
    return net


def build_model(labels):
    net = model()
    loss = get_loss(labels=labels,
                    predictions=net["output"],
                    loss_type=FLAGS.loss_type,
                    scope=FLAGS.loss_type,
                    huber_delta=FLAGS.huber_delta)
    dsc = get_dsc(labels=labels,
                  predictions=net["output"])
    net["loss"] = loss
    net["dsc"] = dsc
    return net
