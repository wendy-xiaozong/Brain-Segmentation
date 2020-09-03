"""
using Kmeans to make the threshold and do crop on the MR image

Some code are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/cropping.py
https://github.com/DM-Berger/autocrop/blob/dec40a194f3ace2d024fd24d8faa503945821015/test/test_masking.py
"""
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import numpy as np
import shutil
from pathlib import Path
from multiprocessing import Pool
from collections import OrderedDict
from sklearn.cluster import KMeans, MiniBatchKMeans
from data.const import cropped_resample_img_folder, cropped_resample_label_folder
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import nibabel as nib
from time import ctime
from tqdm import tqdm
from data.get_subjects import get_subjects
from torch.utils.data import DataLoader

from torchio import DATA, AFFINE
import torchio as tio
from torchio.transforms import (
    Resample,
    Compose
)


def get_numpy_affine(batch):
    img_np, label_np = batch["img"][DATA].squeeze().numpy(), batch["label"][DATA].squeeze().numpy()
    img_affine, label_affine = batch["img"][AFFINE].squeeze().numpy(), batch["label"][AFFINE].squeeze().numpy()
    return img_np, label_np, img_affine, label_affine


def run_resample(batch, img_folder, label_folder) -> int:
    # get the file name
    _, filename = os.path.split(batch["img"]['path'][0])
    filename, _ = os.path.splitext(filename)

    try:
        img, label, img_affine, label_affine = get_numpy_affine(batch)
    except OSError:
        print("OSError! skip file!")
        return

    if img.shape != label.shape:
        print(f"the image: {filename} \n shape {img.shape} is not equal to the label shape {label.shape}")
        return 0

    resample_img_file = nib.Nifti1Image(img, img_affine)
    nib.save(resample_img_file, img_folder / Path(f"{filename}"))
    resample_label_file = nib.Nifti1Image(label, label_affine)
    nib.save(resample_label_file, label_folder / Path(f"{filename}.gz"))
    return 1


def pre_transform() -> Compose:
    transform = Compose([
        Resample(1.0),
    ])
    return transform


if __name__ == "__main__":
    if not os.path.exists(cropped_resample_img_folder):
        os.mkdir(cropped_resample_img_folder)
    if not os.path.exists(cropped_resample_label_folder):
        os.mkdir(cropped_resample_label_folder)

    print(f"{ctime()}: starting ...")

    # for idx, mri in enumerate(get_path(datasets)):
    # if not COMPUTECANADA:
    # run_crop(idx, mri.img_path, mri.label_path, cropped_img_folder, cropped_label_folder)

    idx = 0

    subjects, visual_img_path_list, visual_label_path_list = get_subjects(use_cropped_resampled_data=False)

    transform = pre_transform()
    image_dataset = tio.ImagesDataset(subjects, transform=transform)
    loader = DataLoader(image_dataset,
                        batch_size=1)  # always one because using different label size

    for batch in tqdm(loader):
        idx += run_resample(batch, cropped_resample_img_folder, cropped_resample_label_folder)

    print(f"{ctime()}: ending ...")
    print(f"Totally get {idx} imgs!")
    # show_save_img_and_label(img_2D, label_2D, bbox_percentile_80, bbox_kmeans, "./rectangle_image", idx)
