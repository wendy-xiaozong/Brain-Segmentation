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


def get_CNN_layer(model, number_of_units, channels, name):  # block
    for n in range(number_of_units):
        model = Conv3D(filters=channels, kernel_size=3, strides=1, padding="same", dilation_rate=1, activation='relu',
                       kernel_regularizer=regularizers.l2(0.5), bias_regularizer=regularizers.l2(0.5),
                       name=name + "/conv" + str(n))(model)
        model = BatchNormalization()(model)
    return model


def cnn_3d_segmentation(channels,
                        pool_strides,
                        encoder_units,
                        decoder_units,
                        transition_channels):
    """levels=3,
                              channels=[64, 128, 256],
                              encoder_units=[3, 4, 5],
                              decoder_units=[2, 2],
                              pool_strides=[[2, 2, 2], [1, 2, 2]]"""
    # encoder
    name = 'encoder'
    inputs = Input(shape=(8, 24, 24, 3))  # ??
    conv1 = get_CNN_layer(model=inputs, number_of_units=encoder_units[0], channels=channels[0], name=name + '_level_1')
    conv2 = AveragePooling3D(pool_size=pool_strides[0], strides=pool_strides[0], padding="same")(conv1)
    conv2 = get_CNN_layer(model=conv2, number_of_units=encoder_units[1], channels=channels[1], name=name + '_level_2')
    conv3 = AveragePooling3D(pool_size=pool_strides[1], strides=pool_strides[1], padding="same")(conv2)
    conv3 = get_CNN_layer(model=conv3, number_of_units=encoder_units[2], channels=channels[2], name=name + '_level_3')

    # decoder
    name = 'decoder'
    upconv4 = Conv3DTranspose(filters=channels[1], kernel_size=pool_strides[1], strides=pool_strides[1], padding="same",
                              dilation_rate=1, activation='relu',
                              kernel_regularizer=regularizers.l2(0.5),
                              bias_regularizer=regularizers.l2(0.5), name='decoder_2')(conv3)
    upconv4 = BatchNormalization()(upconv4)
    transition_layer_4 = get_CNN_layer(model=conv2, number_of_units=decoder_units[1], channels=transition_channels[1],
                                       name='transition' + '_level_4')
    merge4 = Concatenate(axis=4)([upconv4, transition_layer_4])
    upconv4 = get_CNN_layer(model=merge4, number_of_units=decoder_units[1], channels=channels[1], name=name + '_level_4')
    upconv5 = Conv3DTranspose(filters=channels[0], kernel_size=pool_strides[0], strides=pool_strides[0], padding="same",
                              dilation_rate=1, activation='relu',
                              kernel_regularizer=regularizers.l2(0.5),
                              bias_regularizer=regularizers.l2(0.5), name='decoder_1')(upconv4)
    upconv5 = BatchNormalization()(upconv5)
    transition_layer_5 = get_CNN_layer(model=conv1, number_of_units=decoder_units[1], channels=transition_channels[1],
                                       name='transition' + '_level_5')
    merge5 = Concatenate(axis=4)([upconv5, transition_layer_5])
    upconv5 = get_CNN_layer(model=merge5, number_of_units=decoder_units[1], channels=channels[1], name=name + '_level_5')
    output = Conv3D(filters=11, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.5),
                    bias_regularizer=regularizers.l2(0.5), name="output")(upconv5)
    conv6 = tf.nn.softmax(logits=output, axis=1, name="softmax")  # ???
    model = Model(inputs, conv6)
    return model


def build_model():
    model = cnn_3d_segmentation(channels=[64, 128, 256],
                                pool_strides=[[2, 2, 2], [1, 2, 2]],
                                encoder_units=[3, 4, 5],
                                decoder_units=[2, 2],
                                transition_channels=[16, 32, 64])
    model.summary()

    model.compile('adam', )
    loss = get_loss(labels=labels,
                    predictions=net["output"],
                    loss_type=FLAGS.loss_type,
                    scope=FLAGS.loss_type,
                    huber_delta=FLAGS.huber_delta)
    dsc = get_dsc(labels=labels,
                  predictions=net["output"])
    net["loss"] = loss
    net["dsc"] = dsc
    return model
