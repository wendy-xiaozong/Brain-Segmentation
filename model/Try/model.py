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
            # initial_out_channels_power=4,  # ?
            initial_out_channels_power=2,
            layers_per_residual_block=2,
            residual_blocks_per_dilation=3,
            dilations=3,
            batch_norm=True,
            instance_norm=False,
            residual=True,
            padding_mode='constant',
            add_dropout_layer=False,
            initialization: Optional[str] = None,
    ):
        assert dimensions in (2, 3)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_residual_block = layers_per_residual_block
        self.residual_blocks_per_dilation = residual_blocks_per_dilation
        self.dilations = dilations

        # List of blocks
        blocks = nn.ModuleList()

        # Add first conv layer
        initial_out_channels = 2 ** initial_out_channels_power
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
        # blocks.append(first_conv_block)

        # Add dilation blocks
        in_channels = out_channels = initial_out_channels
        dilation_block = None  # to avoid pylint errors
        for dilation_idx in range(dilations):
            if dilation_idx >= 1:
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
            out_channels = 80
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
        self.classifier = ConvolutionalBlock(
            in_channels=out_channels + 8,
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

        # blocks.append(classifier)
        self.block = nn.Sequential(*blocks)
        self.softmax = nn.Softmax(dim=1)

        self.unet = UNet(
            in_channels=4,
            num_encoding_blocks=2,
            out_channels_first_layer=8,  # whether this could be used?
            kernal_size=5,
            normalization='InstanceNorm3d',
            module_type="Unet",
            downsampling_type='max',
            dropout=0,
            use_classifier=False,
        )

    def forward(self, x):
        first_layer = self.first_conv_block(x)
        # Unet
        unet_output = self.unet(first_layer)
        # HighResNet
        highResNet_output = x = self.block(first_layer)
        x = torch.cat((unet_output, highResNet_output), dim=1)
        x = self.classifier(x)
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