import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary
# from monai.networks.nets import UNet
from model.unet.unet import UNet
import torch
from model.Try.model import Module
# from model.unet.unet import UNet
from model.highResNet.highresnet import HighResNet


class NewModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 96, 96, 96)

        self.out_classes = 139
        self.deepth = 3
        self.kernal_size = 5  # whether this affect the model to learn?
        self.module_type = 'Unet'
        self.downsampling_type = 'max'
        self.normalization = 'Batch'
        # self.unet = UNet(
        #     in_channels=4,
        #     out_classes=4,
        #     num_encoding_blocks=2,
        #     out_channels_first_layer=8,
        #     kernal_size=3,
        #     normalization=self.normalization,
        #     module_type="Unet",
        #     downsampling_type=self.downsampling_type,
        #     dropout=0,
        #     use_classifier=False,
        # )

        self.myModel = Module(
            in_channels=1,
            out_channels=self.out_classes,
            dimensions=3
        )

    def forward(self, x):
        # return self.unet(x)
        return self.myModel(x)


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
    # HighResNet = HighResNetModel()
    # print("highResNet Model:")
    # print(ModelSummary(HighResNet, mode="full"))

    newNet = NewModel()
    print("newNet Model:")
    print(ModelSummary(newNet, mode="full"))
