from typing import Optional
import torch.nn as nn
from .conv import ConvolutionalBlock
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            dimensions: int,
            num_encoding_blocks: int,
            kernal_size: int,
            module_type: str,
            normalization: str,
            downsampling_type: str,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dropout: float = 0.3,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        # self.dilation = initial_dilation
        is_first_block = True
        for i in range(num_encoding_blocks):  # 3
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                out_channels=in_channels*2,
                dimensions=dimensions,
                normalization=normalization,
                kernal_size=kernal_size,
                module_type=module_type,
                downsampling_type=downsampling_type,
                is_first_block=is_first_block,
                padding=2,
                padding_mode=padding_mode,
                activation=activation,
                dropout=dropout,
                num_block=i,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:  # ?
                in_channels = out_channels_first
                out_channels_first = in_channels * 2

            # dilation is always None
            # if self.dilation is not None:
            #     self.dilation *= 2
            self.out_channels = self.encoding_blocks[-1].out_channels

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    # @property
    # def out_channels(self):
    #     return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            out_channels: int,
            dimensions: int,
            normalization: Optional[str],
            kernal_size: int = 5,
            module_type: str = 'Unet',
            downsampling_type: Optional[str] = 'conv',
            is_first_block: bool = False,
            padding: int = 2,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dropout: float = 0.3,
            num_block: int = 0,
    ):
        super().__init__()

        self.num_block = num_block
        self.module_type = module_type

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            kernal_size=kernal_size,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout,
        )
        # print(f"conv1: in_channels:{in_channels}, out_channels_first:{out_channels_first}")
        out_channels_second = out_channels_first
        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels,
            normalization=normalization,
            kernal_size=kernal_size,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout,
        )
        # print(f"conv2: in_channels:{out_channels_first}, out_channels_first:{out_channels_second}")
        if module_type == 'ResUnet':
            self.conv_residual = ConvolutionalBlock(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels_second,
                kernal_size=1,
                normalization=None,
                activation=None,
            )
        # print(f"residual: in_channels:{in_channels}, out_channels:{out_channels_first}")

        self.downsampling_type = downsampling_type

        self.downsample = None
        if downsampling_type == "max":
            self.downsample = get_downsampling_maxpooling_layer(dimensions, downsampling_type)
        elif downsampling_type == "conv":
            self.downsample = get_downsampling_conv_layer(in_channels=out_channels_second)

        # self.out_channels = self.conv2.conv_layer.out_channels
        self.out_channels = out_channels_second

    def forward(self, x):
        if self.module_type == "ResUnet":
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            # print(f"connection shape: {connection.shape}")
            # print(f"x shape: {x.shape}")
            x += connection
        elif self.module_type == "Unet":
            x = self.conv1(x)
            x = self.conv2(x)

        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
        return x, skip_connection


def get_downsampling_maxpooling_layer(
        dimensions: int,
        pooling_type: str,
        kernel_size: int = 2,
        stride: int = 2,
) -> nn.Module:
    class_name = '{}Pool{}d'.format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)


def get_downsampling_conv_layer(
        in_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
) -> nn.Module:
    class_name = 'Conv3d'
    class_ = getattr(nn, class_name)
    return class_(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride)
