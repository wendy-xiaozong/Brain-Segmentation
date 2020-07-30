from typing import Union, List
import pytorch_lightning as pl
from torchio import DATA
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data.get_subjects import get_subjects
from data.const import CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1, COMPUTECANADA
from data.transform import get_train_transforms, get_val_transform, get_test_transform
from argparse import ArgumentParser
from data.const import SIZE
from model.unet.unet import UNet
from utils.matrix import get_score
from utils.loss import dice_loss
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from postprocess.visualize import log_all_info
from torch import Tensor
import torchio
import torch
import random


class Lightning_Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Lightning_Unet, self).__init__()
        self.hparams = hparams
        self.out_classes = 139
        self.unet = UNet(
            in_channels=1,
            out_classes=self.out_classes,
            num_encoding_blocks=3,
            out_channels_first_layer=32,
            normalization="Group",
            upsampling_type='conv',
            padding=2,
            dropout=0,
        )
        self.lr = 1e-3

        # torchio parameters
        # need to try to find the suitable value
        self.max_queue_length = 100
        self.patch_size = 96  # from the HighResnet
        self.samples_per_volume = 10

        self.subjects = get_subjects()
        random.seed(42)
        random.shuffle(self.subjects)  # shuffle it to pick the val set
        num_subjects = len(self.subjects)
        num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training
        self.training_subjects = self.subjects[:num_training_subjects]
        self.validation_subjects = self.subjects[num_training_subjects:]

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x)

    # Called at the beginning of fit and test. This is a good hook when you need to build models dynamically or
    # adjust something about them. This hook is called on every process when using DDP.
    def setup(self, stage):
        self.subjects = get_subjects()
        random.seed(42)
        random.shuffle(self.subjects)  # shuffle it to pick the val set
        num_subjects = len(self.subjects)
        num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training
        self.training_subjects = self.subjects[:num_training_subjects]
        self.validation_subjects = self.subjects[num_training_subjects:]

    def train_dataloader(self) -> DataLoader:
        training_transform = get_train_transforms()
        train_imageDataset = torchio.ImagesDataset(self.training_subjects, transform=training_transform)

        patches_training_set = torchio.Queue(
            subjects_dataset=train_imageDataset,
            # Maximum number of patches that can be stored in the queue.
            # Using a large number means that the queue needs to be filled less often,
            # but more CPU memory is needed to store the patches.
            max_length=self.max_queue_length,
            # Number of patches to extract from each volume.
            # A small number of patches ensures a large variability in the queue,   ??? how to understand this??
            # but training will be slower.
            samples_per_volume=self.samples_per_volume,
            #  A sampler used to extract patches from the volumes.
            sampler=torchio.sampler.UniformSampler(self.patch_size),
            num_workers=10,
            # If True, the subjects dataset is shuffled at the beginning of each epoch,
            # i.e. when all patches from all subjects have been processed
            shuffle_subjects=False,
            # If True, patches are shuffled after filling the queue.
            shuffle_patches=True,
            verbose=True,
        )

        training_loader = DataLoader(patches_training_set,
                                     batch_size=self.hparams.batch_size)

        print('Training set:', len(train_imageDataset), 'subjects')
        return training_loader

    def val_dataloader(self) -> DataLoader:
        val_transform = get_val_transform()
        val_imageDataset = torchio.ImagesDataset(self.validation_subjects, transform=val_transform)

        patches_validation_set = torchio.Queue(
            subjects_dataset=val_imageDataset,
            max_length=self.max_queue_length,
            samples_per_volume=self.samples_per_volume,
            sampler=torchio.sampler.UniformSampler(self.patch_size),
            num_workers=10,
            shuffle_subjects=False,
            shuffle_patches=True,
            verbose=True,
        )

        val_loader = DataLoader(patches_validation_set,
                                batch_size=self.hparams.batch_size * 2)
        print('Validation set:', len(val_loader), 'subjects')
        return val_loader

    def test_dataloader(self):
        test_transform = get_test_transform()
        # using all the data to test
        test_imageDataset = torchio.ImagesDataset(self.subjects, transform=test_transform)
        test_loader = DataLoader(test_imageDataset,
                                 batch_size=1)  # always one because using different label size
                                 # num_workers=8)
        print('Testing set:', len(test_imageDataset), 'subjects')
        return test_loader

    # need to adding more things
    def configure_optimizers(self):
        # Setting up the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = MultiStepLR(optimizer, milestones=[1, 10], gamma=0.1)
        return optimizer

    def prepare_batch(self, batch):
        inputs, targets = batch["img"][DATA], batch["label"][DATA]
        if torch.isnan(inputs).any():
            print("there is nan in input data!")
            inputs[inputs != inputs] = 0
        if torch.isnan(targets).any():
            print("there is nan in targets data!")
            targets[targets != targets] = 0
        # making the label as binary, it is very strange because if the label is not binary
        # the whole model cannot learn at all
        # target_one_hot = F.one_hot(targets)
        return inputs, targets

    def training_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        # print(f"training input range: {torch.min(inputs)} - {torch.max(inputs)}")
        probs = self(inputs)
        print(f"probs: {probs.shape}")
        dice, iou, _, _ = get_score(probs, targets)
        loss = dice_loss(input=probs, target=targets, to_onehot=True)
        # if batch_idx != 0 and ((self.current_epoch >= 1 and dice.item() < 0.5) or batch_idx % 100 == 0):
        #     input = inputs.chunk(inputs.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     target = targets.chunk(targets.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     prob = probs.chunk(probs.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     # really have problem in there, need to fix it
        #     # dice_score, _, _, _ = get_score(torch.unsqueeze(prob, 0), torch.unsqueeze(target, 0))
        #     log_all_info(self, input, target, prob, batch_idx, "training")
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        tensorboard_logs = {"train_loss": loss, "train_IoU": iou, "train_dice": dice}
        return {'loss': loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_id):
        inputs, targets = self.prepare_batch(batch)
        probs = self(inputs)
        # print(f"probs: {probs.shape}")
        loss = dice_loss(input=probs, target=targets, to_onehot=True)
        dice, iou, sensitivity, specificity = get_score(probs, targets)
        return {'val_step_loss': loss,
                'val_step_dice': dice,
                'val_step_IoU': iou,
                "val_step_sensitivity": sensitivity,
                "val_step_specificity": specificity
                }

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        avg_val_dice = torch.stack([x['val_step_dice'] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": outputs[0]['val_step_loss'],  # the outputs is a dict wrapped in a list
            "val_dice": outputs[0]['val_step_dice'],
            "val_IoU": outputs[0]['val_step_IoU'],
            "val_sensitivity": outputs[0]['val_step_sensitivity'],
            "val_specificity": outputs[0]['val_step_specificity']
        }
        return {"loss": avg_loss, "val_loss": avg_loss, "val_dice": avg_val_dice, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        # print(f"training input range: {torch.min(inputs)} - {torch.max(inputs)}")
        logits = self(inputs)
        logits = F.interpolate(logits, size=logits.size()[2:])
        probs = torch.sigmoid(logits)
        dice, iou, _, _ = get_score(probs, targets)
        if batch_idx != 0 and batch_idx % 50 == 0:  # save total about 10 picture
            input = inputs.chunk(inputs.size()[0], 0)[0]  # split into 1 in the dimension 0
            target = targets.chunk(targets.size()[0], 0)[0]  # split into 1 in the dimension 0
            logit = probs.chunk(logits.size()[0], 0)[0]  # split into 1 in the dimension 0
            # need to add the dice score here
            # log_all_info(self, input, target, logit, batch_idx, "testing")
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = dice_loss(probs, targets)
        dice, iou, sensitivity, specificity = get_score(probs, targets)
        return {'test_step_loss': loss,
                'test_step_dice': dice,
                'test_step_IoU': iou,
                'test_step_sensitivity': sensitivity,
                'test_step_specificity': specificity
                }

    def test_epoch_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['test_step_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['test_step_dice'] for x in outputs]).mean()
        avg_IoU = torch.stack([x['test_step_IoU'] for x in outputs]).mean()
        avg_sensitivity = torch.stack([x['test_step_sensitivity'] for x in outputs]).mean()
        avg_specificity = torch.stack([x['test_step_specificity'] for x in outputs]).mean()
        tensorboard_logs = {
            "avg_test_loss": avg_loss.item(),  # the outputs is a dict wrapped in a list
            "avg_test_dice": avg_dice.item(),
            "avg_test_IoU": avg_IoU.item(),
            "avg_test_sensitivity": avg_sensitivity.item(),
            "avg_test_specificity": avg_specificity.item(),
        }
        return {'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        parameters defined here will be available to the model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=1, help='Batch size', dest='batch_size')
        parser.add_argument("--learning_rate", type=float, default=1e-3, help='Learning rate')
        # parser.add_argument("--normalization", type=str, default='Group', help='the way of normalization')
        parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        parser.add_argument("--loss", type=str, default="BCEWL", help='Loss Function')
        return parser
