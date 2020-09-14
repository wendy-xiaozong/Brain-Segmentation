# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional
import torch.nn as nn
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock

__all__ = ['UNet', 'UNet2D', 'UNet3D']


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 1,
            dimensions: int = 3,
            num_encoding_blocks: int = 3,
            out_channels_first_layer: int = 8,
            kernal_size: int = 5,
            module_type: str = 'Unet',
            normalization: str = 'Group',
            downsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0.3,
            monte_carlo_dropout: float = 0.3,
            use_classifier: bool = True,
            ):
        super().__init__()
        depth = num_encoding_blocks  # 3
        self.use_classifier = use_classifier

        # Force padding if residual blocks
        # Why this???? How this do?
        if residual:
            padding = 1

        # Encoder Conv3D *2 *2
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels_first=out_channels_first_layer,
            dimensions=dimensions,
            num_encoding_blocks=depth,
            kernal_size=kernal_size,
            module_type=module_type,
            normalization=normalization,
            downsampling_type=downsampling_type,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout,
        )

        # There is only one layer at the bottom of the Unet
        in_channels = self.encoder.out_channels
        in_channels_skip_connection = in_channels
        out_channels_first = in_channels * 2
        self.bottom_block = EncodingBlock(
            in_channels=in_channels,
            out_channels_first=out_channels_first,
            out_channels=in_channels,
            dimensions=dimensions,
            normalization=normalization,
            kernal_size=kernal_size,
            module_type=module_type,
            downsampling_type=None,
            padding=2,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout,
        )

        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type="conv",
            num_decoding_blocks=num_decoding_blocks,
            kernal_size=kernal_size,
            module_type=module_type,
            normalization=normalization,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        # self.monte_carlo_layer = None
        # if monte_carlo_dropout:
        #     dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
        #     self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_channels=out_classes,
            kernal_size=1, activation=None, normalization=None,
            dropout=0,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        # print(f"first skip connection shape: {skip_connections[0].shape}")
        x = self.bottom_block(encoding)
        # print(f"bottom block shape: {x.shape}")
        x = self.decoder(skip_connections, x)
        if self.use_classifier:
            x = self.classifier(x)
            return self.softmax(x)
        else:
            return x


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