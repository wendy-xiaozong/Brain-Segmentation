# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional
import torch
import torch.nn as nn
from model.unet.unet import UNet
from model.highResNet.dilation import DilationBlock
from model.highResNet.convolution import ConvolutionalBlock


class Module(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dimensions=None,
            initial_out_channels_power=4,
            # initial_out_channels_power=2,
            layers_per_residual_block=2,
            residual_blocks_per_dilation=3,
            dilations=3,
            batch_norm=True,
            instance_norm=False,
            residual=True,
            padding_mode='constant',
            add_dropout_layer=False,  # Why this is False?
            initialization: Optional[str] = None,
    ):
        assert dimensions in (2, 3)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_residual_block = layers_per_residual_block
        self.residual_blocks_per_dilation = residual_blocks_per_dilation
        self.dilations = dilations

        # Add first conv layer
        initial_out_channels = 2 ** initial_out_channels_power
        # only one convolution layer in this block,
        # it is strange because it do not use the preactivation here
        # but in the paper it used
        self.first_conv_block = ConvolutionalBlock(
            in_channels=self.in_channels,
            out_channels=initial_out_channels,
            dilation=1,
            dimensions=dimensions,
            batch_norm=batch_norm,
            instance_norm=instance_norm,
            preactivation=False,
            padding_mode=padding_mode,
        )

        # Add dilation blocks
        in_channels = out_channels = initial_out_channels
        dilation_block = None  # to avoid pylint errors
        # get the first conv block
        self.first_dilated_block = DilationBlock(
            in_channels,
            out_channels,
            dilation=1,
            dimensions=dimensions,
            layers_per_block=layers_per_residual_block,
            num_residual_blocks=residual_blocks_per_dilation,
            batch_norm=batch_norm,
            instance_norm=instance_norm,
            residual=residual,
            padding_mode=padding_mode,
        )
        out_channels *= 2

        # List of blocks
        blocks = nn.ModuleList()
        # The rest conv block
        for dilation_idx in range(1, dilations):
            # need to change this
            if dilation_idx == 1:
                out_channels = in_channels = 32
            else:
                in_channels = dilation_block.out_channels
            dilation = 2 ** dilation_idx
            dilation_block = DilationBlock(
                in_channels,
                out_channels,
                dilation,
                dimensions,
                layers_per_block=layers_per_residual_block,
                num_residual_blocks=residual_blocks_per_dilation,
                batch_norm=batch_norm,
                instance_norm=instance_norm,
                residual=residual,
                padding_mode=padding_mode,
            )
            blocks.append(dilation_block)
            out_channels *= 2
        out_channels = out_channels // 2

        # Add dropout layer as in NiftyNet
        if add_dropout_layer:
            in_channels = out_channels
            out_channels = 80  # What is this??
            dropout_conv_block = ConvolutionalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=1,
                dimensions=dimensions,
                batch_norm=batch_norm,
                instance_norm=instance_norm,
                preactivation=False,
                kernel_size=1,
            )
            blocks.append(dropout_conv_block)
            blocks.append(nn.Dropout3d())

        # Add classifier
        classifier = ConvolutionalBlock(
            in_channels=out_channels,
            out_channels=self.out_channels,
            dilation=1,
            dimensions=dimensions,
            batch_norm=batch_norm,
            instance_norm=instance_norm,
            preactivation=False,
            kernel_size=1,
            activation=False,
            padding_mode=padding_mode,
        )

        blocks.append(classifier)
        self.block = nn.Sequential(*blocks)
        self.softmax = nn.Softmax(dim=1)

        self.unet = UNet(
            in_channels=16,
            out_classes=16,
            num_encoding_blocks=2,
            out_channels_first_layer=32,
            kernal_size=3,
            normalization="Batch",
            module_type="Unet",
            downsampling_type="max",
            dropout=0,
            use_classifier=False,
        )

    def forward(self, x):
        first_layer = self.first_conv_block(x)
        # print(f"first layer shape: {first_layer.shape}")
        # mini-Unet and first part of the highResNet
        unet_output = self.unet(first_layer)
        # print(f"unet output shape: {unet_output.shape}")
        highResNet_first_conv_block = self.first_dilated_block(first_layer)
        x = torch.cat((unet_output, highResNet_first_conv_block), dim=1)
        x = self.block(x)
        return self.softmax(x)


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 2
        kwargs['num_encoding_blocks'] = 5
        kwargs['out_channels_first_layer'] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 3
        kwargs['num_encoding_blocks'] = 3  # 4
        kwargs['out_channels_first_layer'] = 8
        # kwargs['normalization'] = 'batch'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)