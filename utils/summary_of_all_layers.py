import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary
import torch
from model.unet.unet import UNet
from model.highResNet.highresnet import HighResNet


class MyUnetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 96, 96, 96)

        self.out_classes = 139
        self.deepth = 4
        self.kernal_size = 5  # whether this affect the model to learn?
        self.module_type = 'Unet'
        self.downsampling_type = 'max'
        self.normalization = 'InstanceNorm3d'
        self.unet = UNet(
            in_channels=1,
            out_classes=self.out_classes,
            num_encoding_blocks=self.deepth,
            out_channels_first_layer=32,
            kernal_size=self.kernal_size,
            normalization=self.normalization,
            module_type=self.module_type,
            downsampling_type=self.downsampling_type,
            dropout=0,
        )

    def forward(self, x):
        return self.unet(x)


class HighResNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 96, 96, 96)

        self.unet = HighResNet(
            in_channels=1,
            out_channels=139,
            dimensions=3
        )

    def forward(self, x):
        return self.unet(x)


if __name__ == "__main__":
    myUnet = MyUnetModel()
    print("myUnet Model:")
    print(ModelSummary(myUnet, mode="full"))

    highResNet = HighResNetModel()
    print("highResNet Model:")
    print(ModelSummary(highResNet, mode="full"))
