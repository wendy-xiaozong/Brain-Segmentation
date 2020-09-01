import pytorch_lightning as pl
from torchio import DATA, PATH
from torch.utils.data import DataLoader
from data.get_subjects import get_subjects
from data.const import COMPUTECANADA
from data.transform import get_train_transforms, get_val_transform, get_test_transform
from argparse import ArgumentParser
from model.unet.unet import UNet
from model.highResNet.highresnet import HighResNet
from utils.matrix import get_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import inspect
import torch.nn.functional as F
from postprocess.visualize import log_all_info
from torch import Tensor
from time import ctime
from monai.losses import DiceLoss
# from utils.gpu_mem_track import MemTracker
import torchio
import torch
import random
import os
from pathlib import Path
import pandas as pd
from pytorch_lightning.metrics.functional import to_onehot
from utils.enums import LossReduction

import gc
import datetime
import pynvml
import numpy as np


class Lightning_Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Lightning_Unet, self).__init__()
        self.hparams = hparams

        self.out_classes = 139
        self.deepth = self.hparams.deepth
        self.kernel_size = self.hparams.kernel_size
        self.module_type = 'Unet'
        self.downsampling_type = 'max'
        self.normalization = 'InstanceNorm3d'
        if self.hparams.model == "unet":
            self.unet = UNet(
                in_channels=1,
                out_classes=self.out_classes,
                num_encoding_blocks=self.deepth,
                out_channels_first_layer=self.hparams.out_channels_first_layer,
                kernal_size=self.kernel_size,
                normalization=self.normalization,
                module_type=self.module_type,
                downsampling_type=self.downsampling_type,
                dropout=0,
            )
        elif self.hparams.model == "highResNet":
            self.unet = HighResNet(
                in_channels=1,
                out_channels=self.out_classes,
                dimensions=3
            )

        # torchio parameters
        # ?need to try to find the suitable value
        self.max_queue_length = 10
        self.patch_size = self.hparams.patch_size
        # Number of patches to extract from each volume. A small number of patches ensures a large variability
        # in the queue, but training will be slower.
        self.samples_per_volume = 5
        self.num_workers = 0
        if not self.hparams.include_background:
            print("It is not included the background.")

        if not COMPUTECANADA:
            self.max_queue_length = 10
            self.patch_size = 48
            self.num_workers = 8
            self.subjects, self.visual_img_path_list, self.visual_label_path_list = get_subjects(
                use_cropped_data=True
            )
            random.seed(42)
            random.shuffle(self.subjects)  # shuffle it to pick the val set
            num_subjects = len(self.subjects)
            num_training_subjects = int(num_subjects * 0.995)  # using only around 25 images
            self.training_subjects = self.subjects[:num_training_subjects]
            self.validation_subjects = self.subjects[num_training_subjects:]

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x)

    # Called at the beginning of fit and test. This is a good hook when you need to build models dynamically or
    # adjust something about them. This hook is called on every process when using DDP.
    def setup(self, stage):
        self.subjects, self.visual_img_path_list, self.visual_label_path_list = get_subjects(
            use_cropped_data=self.hparams.use_cropped_img)
        random.seed(42)
        random.shuffle(self.subjects)  # shuffle it to pick the val set
        num_subjects = len(self.subjects)
        num_training_subjects = int(num_subjects * 0.99)  # using only around 25 images
        self.training_subjects = self.subjects[:num_training_subjects]
        self.validation_subjects = self.subjects[num_training_subjects:]
        self.test_subjects = self.subjects
        self.val_times = 0
        self.test_times = 0
        self.df = pd.DataFrame(columns=['filename'])

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
            # A small number of patches ensures a large variability in the queue,
            # but training will be slower.
            samples_per_volume=self.samples_per_volume,
            #  A sampler used to extract patches from the volumes.
            sampler=torchio.sampler.UniformSampler(self.patch_size),
            num_workers=self.num_workers,
            # If True, the subjects dataset is shuffled at the beginning of each epoch,
            # i.e. when all patches from all subjects have been processed
            shuffle_subjects=False,
            # If True, patches are shuffled after filling the queue.
            shuffle_patches=True,
            verbose=True,
        )

        training_loader = DataLoader(patches_training_set,
                                     batch_size=self.hparams.batch_size)

        print(f"{ctime()}: getting number of training subjects {len(training_loader)}")
        return training_loader

    def val_dataloader(self) -> DataLoader:
        val_imageDataset = torchio.ImagesDataset(self.validation_subjects)

        # patches_validation_set = torchio.Queue(
        #     subjects_dataset=val_imageDataset,
        #     max_length=self.max_queue_length,
        #     samples_per_volume=self.samples_per_volume,
        #     sampler=torchio.sampler.UniformSampler(self.patch_size),
        #     num_workers=self.num_workers,
        #     shuffle_subjects=False,
        #     shuffle_patches=True,
        #     verbose=True,
        # )

        # the batch_size here only could be 1 because we only could handle one image to aggregate
        val_loader = DataLoader(val_imageDataset, batch_size=1)
        print(f"{ctime()}: getting number of validation subjects {len(val_loader)}")
        return val_loader

    def test_dataloader(self):
        test_imageDataset = torchio.ImagesDataset(self.test_subjects)

        # patches_validation_set = torchio.Queue(
        #     subjects_dataset=val_imageDataset,
        #     max_length=self.max_queue_length,
        #     samples_per_volume=self.samples_per_volume,
        #     sampler=torchio.sampler.UniformSampler(self.patch_size),
        #     num_workers=self.num_workers,
        #     shuffle_subjects=False,
        #     shuffle_patches=True,
        #     verbose=True,
        # )

        # the batch_size here only could be 1 because we only could handle one image to aggregate
        test_loader = DataLoader(test_imageDataset, batch_size=1)
        print(f"{ctime()}: getting number of validation subjects {len(test_loader)}")
        return test_loader

    # need to adding more things
    def configure_optimizers(self):
        # Setting up the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = MultiStepLR(optimizer, milestones=[1, 10], gamma=0.1)
        # need to find what other used here
        lr_dict = {
                'scheduler': ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5,
                                               patience=10, min_lr=1e-6),
                # might need to change here
                'monitor': 'val_checkpoint_on',  # Default: val_loss
                'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler, default
                # 'interval': 'step',
                'interval': 'epoch',
                # need to change here
                # 'frequency': 300
                'frequency': 1
            }

        return [optimizer], [lr_dict]

    def prepare_batch(self, batch):
        inputs, targets = batch["img"][DATA], batch["label"][DATA]

        # print the path
        # img_path, label_path = batch["img"][PATH], batch["label"][PATH]
        # print(f"img path: {img_path}")
        # print(f"label path: {label_path}")

        if torch.isnan(inputs).any():
            print("there is nan in input data!")
            inputs[inputs != inputs] = 0
        if torch.isnan(targets).any():
            print("there is nan in targets data!")
            targets[targets != targets] = 0
        return inputs, targets

    def training_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        pred = self(inputs)
        # diceloss = DiceLoss(include_background=True, to_onehot_y=True)
        # loss = diceloss.forward(input=probs, target=targets)
        # dice, iou, _, _ = get_score(batch_preds, batch_targets, include_background=True)
        # gdloss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True)
        # loss = gdloss.forward(input=batch_preds, target=batch_targets)
        # if batch_idx != 0 and ((self.current_epoch >= 1 and dice.item() < 0.5) or batch_idx % 100 == 0):
        #     input = inputs.chunk(inputs.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     target = targets.chunk(targets.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     prob = probs.chunk(probs.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     ＃　really have problem in there, need to fix it
        #     dice_score, _, _, _ = get_score(torch.unsqueeze(prob, 0), torch.unsqueeze(target, 0))
        #     log_all_info(self, input, target, prob, batch_idx, "training", dice_score.item())
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        diceloss = DiceLoss(include_background=self.hparams.include_background, to_onehot_y=True)
        loss = diceloss.forward(input=pred, target=targets)
        # What is the loos I need to set here? when I am using the batch size?

        # gdloss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True)
        # loss = gdloss.forward(input=batch_preds, target=batch_targets)

        # the loss for prog_bar is not corrected, is there anything I write wrong?
        result = pl.TrainResult(minimize=loss)
        # logs metrics for each training_step, to the progress bar and logger
        result.log("train_loss", loss, prog_bar=True, sync_dist=True, logger=True, reduce_fx=torch.mean, on_step=True,
                   on_epoch=False)
        # we cannot compute the matrixs on the patches, because they do not contain all the 138 segmentations
        # So they would return 0 on some of the classes, making the matrixs not accurate
        return result

    # It supports only need when using DP or DDP2, I should not need it because I am using ddp
    # but I have some problem with the dice score, So I am just trying ...
    # def training_step_end(self, outputs) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
    #     print(f"outputs shape: {outputs['train_step_preds'].shape}")
    #     batch_preds = torch.stack([x['train_step_preds'] for x in outputs])
    #     batch_targets = torch.stack([x['train_step_target'] for x in outputs])
    # dice, iou, _, _ = get_score(batch_preds, batch_targets, include_background=True)
    # dice = dice_score(pred=batch_preds, target=batch_targets, bg=True)

    def compute_from_aggregating(self, input, target, if_path: bool, type_as_tensor=None, whether_to_return_img=False):
        if if_path:
            cur_img_subject = torchio.Subject(
                img=torchio.Image(input, type=torchio.INTENSITY)
            )
            cur_label_subject = torchio.Subject(
                img=torchio.Image(target, type=torchio.LABEL)
            )
        else:
            cur_img_subject = torchio.Subject(
                img=torchio.Image(tensor=input.squeeze().cpu().detach(), type=torchio.INTENSITY)
            )
            cur_label_subject = torchio.Subject(
                img=torchio.Image(tensor=target.squeeze().cpu().detach(), type=torchio.LABEL)
            )

        # This is different? why?
        # print(f"before transform input: {cur_img_subject.img.data.shape}")
        # print(f"before transform label: {cur_label_subject.img.data.shape}")

        transform = get_val_transform()
        preprocessed_img = transform(cur_img_subject)
        preprocessed_label = transform(cur_label_subject)

        # print(f"after transform input: {preprocessed_img.img.data.shape}")
        # print(f"after transform label: {preprocessed_label.img.data.shape}")

        patch_overlap = self.hparams.patch_overlap  # is there any constrain?
        grid_sampler = torchio.inference.GridSampler(
            preprocessed_img,
            self.patch_size,
            patch_overlap,
        )

        patch_loader = torch.utils.data.DataLoader(grid_sampler)
        aggregator = torchio.inference.GridAggregator(grid_sampler)

        for patches_batch in patch_loader:
            input_tensor = patches_batch['img'][torchio.DATA]
            # used to convert tensor to CUDA
            if not if_path:
                input_tensor = input_tensor.type_as(input)
            else:
                input_tensor = input_tensor.type_as(type_as_tensor['val_dice'])
            locations = patches_batch[torchio.LOCATION]
            preds = self(input_tensor)  # use cuda
            labels = preds.argmax(dim=torchio.CHANNELS_DIMENSION, keepdim=True)  # use cuda
            aggregator.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()  # not using cuda!!!!

        if if_path or whether_to_return_img:
            return preprocessed_img.img.data, output_tensor, preprocessed_label.img.data
        else:
            return output_tensor, preprocessed_label.img.data

    def validation_step(self, batch, batch_id):
        input, target = self.prepare_batch(batch)
        output_tensor, target_tensor = self.compute_from_aggregating(input, target, if_path=False)  # in CPU

        # pred = self(inputs)
        # gdloss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True)
        # loss = gdloss.forward(input=probs, target=targets)

        diceloss = DiceLoss(include_background=self.hparams.include_background, to_onehot_y=True)
        output_tensor = to_onehot(output_tensor, num_classes=139)
        loss = diceloss.forward(input=output_tensor, target=target_tensor.unsqueeze(dim=1))  # all in CPU
        loss_cuda = loss.type_as(input)
        output_tensor_cuda = output_tensor.type_as(input)
        target_tensor_cuda = target_tensor.type_as(input)
        del output_tensor, target_tensor, loss, input, target
        # dice, iou, sensitivity, specificity = get_score(output_tensor_cuda, target_tensor_cuda,
        #                                                 include_background=True, reduction=LossReduction.NONE)
        dice, iou, sensitivity, specificity = get_score(output_tensor_cuda, target_tensor_cuda,
                                                        include_background=True)

        result = pl.EvalResult(early_stop_on=dice, checkpoint_on=dice)
        result.log('val_loss', loss_cuda, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        # why I have this error?
        # ValueError: only one element tensors can be converted to Python scalars
        # When I test it, I don't have it
        result.log('val_dice', dice, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        result.log('val_IoU', iou, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        result.log('val_sensitivity', sensitivity, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        result.log('val_specificity', specificity, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        return result

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, validation_step_output_result):
        # visualization part
        cur_img_path = self.visual_img_path_list[self.val_times % len(self.visual_img_path_list)]
        cur_label_path = self.visual_label_path_list[self.val_times % len(self.visual_label_path_list)]

        img, output_tensor, target_tensor = self.compute_from_aggregating(cur_img_path, cur_label_path,
                                                                          if_path=True,
                                                                          type_as_tensor=validation_step_output_result)
        # print(f"validation_epoch_end_output_tensor: {output_tensor.requires_grad}")
        # print(f"validation_epoch_end_target_tensor: {target_tensor.requires_grad}")
        output_tensor_cuda = output_tensor.type_as(validation_step_output_result['val_dice'])
        target_tensor_cuda = target_tensor.type_as(validation_step_output_result['val_dice'])
        del output_tensor, target_tensor
        # using CUDA
        dice, iou, sensitivity, specificity = get_score(pred=output_tensor_cuda, target=target_tensor_cuda,
                                                        include_background=True)

        log_all_info(self,
                     img,
                     target_tensor_cuda,
                     output_tensor_cuda,
                     dice,
                     self.val_times, filename=None)
        self.val_times += 1

        return validation_step_output_result

    def test_step(self, batch, batch_idx):
        input, target = self.prepare_batch(batch)
        img, output_tensor, target_tensor = self.compute_from_aggregating(input, target, if_path=False,
                                                                          whether_to_return_img=True)  # in CPU

        # pred = self(inputs)
        # gdloss = GeneralizedDiceLoss(include_background=True, to_onehot_y=True)
        # loss = gdloss.forward(input=probs, target=targets)

        output_tensor_cuda = output_tensor.type_as(input)
        target_tensor_cuda = target_tensor.type_as(input)
        del output_tensor, target_tensor, input, target
        # dice, iou, sensitivity, specificity = get_score(output_tensor_cuda, target_tensor_cuda,
        #                                                 include_background=True, reduction=LossReduction.NONE)
        dice, iou, sensitivity, specificity = get_score(pred=output_tensor_cuda, target=target_tensor_cuda,
                                                        include_background=True)
        # if dice.item() < 0.5:
        #     # get path of img and target
        #     img_path, label_path = batch["img"][PATH][0], batch["label"][PATH][0]
        #     # move the deleted file to the folder
        #     os.system(f"mv {img_path} {delete_img_folder}")
        #     os.system(f"mv {label_path} {delete_label_folder}")
        #     # get the filename
        #     _, filename = os.path.split(img_path)
        #     # need to add the filename into the tensorboard
        #     log_all_info(self,
        #                  img,
        #                  target_tensor_cuda,
        #                  output_tensor_cuda,
        #                  dice,
        #                  self.test_times,
        #                  filename=filename)
        #     self.test_times += 1
        #     # add the filename into the dataframe.
        #     self.df.loc[self.df.shape[0]] = {"filename": filename}

        result = pl.EvalResult()
        result.log('test_dice', dice, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        result.log('test_IoU', iou, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        result.log('test_sensitivity', sensitivity, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        result.log('test_loss', specificity, on_step=False, on_epoch=True, logger=True, prog_bar=False,
                   reduce_fx=torch.mean, sync_dist=True)
        return result

    # def test_epoch_end(self, outputs):
    #     # torch.stack: Concatenates sequence of tensors along a new dimension.
    #     tensorboard_logs = {
    #         "test_dice": torch.stack([x['val_step_dice'] for x in outputs]).mean(),
    #         "test_IoU": torch.stack([x['val_step_IoU'] for x in outputs]).mean(),
    #         "test_sensitivity": torch.stack([x['val_step_sensitivity'] for x in outputs]).mean(),
    #         "test_specificity": torch.stack([x['val_step_specificity'] for x in outputs]).mean()
    #     }
    #     # save the file
    #     # self.df.to_csv(Path(__file__).resolve().parent.parent.parent / f"run:{self.hparams.run}-deleted_data.csv")
    #     return {'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        parameters defined here will be available to the model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=1, help='Batch size', dest='batch_size')
        # From the generalizedDiceLoss paper
        parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate')
        # parser.add_argument("--normalization", type=str, default='Group', help='the way of normalization')
        parser.add_argument("--down_sample", type=str, default="max", help='the way to down sample')
        parser.add_argument("--loss", type=str, default="BCEWL", help='Loss Function')
        parser.add_argument("--model", type=str, default="unet", help='to specify which model to choose')
        parser.add_argument("--out_channels_first_layer", type=int, default=32, help="the first layer's out channels")
        parser.add_argument("--run", type=int, default=1, help="number of running times")
        parser.add_argument("--include_background", action="store_true",
                            help='whether include background to compute the dice loss and score')
        parser.add_argument("--use_cropped_img", action="store_true", help='whether use the cropped image')
        parser.add_argument("--deepth", type=int, default=1, help="the deepth of the unet")
        parser.add_argument("--kernel_size", type=int, default=3, help="the kernal size")
        parser.add_argument("--patch_size", type=int, default=96, help="the patch size")
        parser.add_argument("--patch_overlap", type=int, default=10)
        return parser
