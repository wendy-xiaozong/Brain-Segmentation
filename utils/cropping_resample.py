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
from data.const import COMPUTECANADA, cropped_img_folder, cropped_label_folder
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


# have similar outcome to the kmeans, but kmeans have dramtically better result on some images
def create_nonzero_mask_percentile_80(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = (data > np.percentile(data, 70))
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def create_nonzero_mask_kmeans(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    flat = data.ravel()  # Return a contiguous flattened array.

    # using 1 dimension kMeans here to compute the thresholds
    # using code from
    # https://github.com/DM-Berger/autocrop/blob/dec40a194f3ace2d024fd24d8faa503945821015/test/test_masking.py#L19-L25
    # n_job=The number of OpenMP threads to use for the computation.
    # batch_size controls the number of randomly selected observations in each batch.
    # The larger the the size of the batch, the more computationally costly the training process.
    # Increasing the batch size may also help avoid reassignment triggering by some clusters becoming to small
    # just from sampling variation.
    km = MiniBatchKMeans(n_clusters=4, batch_size=1000).fit(flat.reshape(-1, 1))
    # km = KMeans(n_clusters=4, n_jobs=1).fit(flat.reshape(-1, 1))  # more slowly
    gs = [km.labels_ == i for i in range(4)]
    maxs = sorted([np.max(flat[g]) for g in gs])  # choose the max value in the min group
    thresh = maxs[0]

    this_mask = (data > thresh)
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_to_nonzero(data, seg):
    """
    :param data:
    :param seg: label image
    :return:
    """
    # nonzero_mask_percentile_80 = create_nonzero_mask_percentile_80(data)
    nonzero_mask_kmeans = create_nonzero_mask_kmeans(data)
    # bbox_percentile_80 = get_bbox_from_mask(nonzero_mask_percentile_80, 0)
    bbox_kmeans = get_bbox_from_mask(nonzero_mask_kmeans, 0)

    data = crop_to_bbox(data, bbox_kmeans)
    seg = crop_to_bbox(seg, bbox_kmeans)
    return data, seg


def get_numpy_affine(batch):
    img_np, label_np = batch["img"][DATA].squeeze().numpy(), batch["label"][DATA].squeeze().numpy()
    img_affine, label_affine = batch["img"][AFFINE].squeeze().numpy(), batch["label"][AFFINE].squeeze().numpy()
    return img_np, label_np, img_affine, label_affine


def show_save_img_and_label(img_2D, label_2D, bbox_percentile_80, bbox_kmeans, path, idx):
    img_2D = np.where(label_2D > 0.5, np.max(img_2D), img_2D)
    plt.imshow(img_2D)
    current_axis = plt.gca()
    rect1 = patches.Rectangle((bbox_percentile_80[1][0], bbox_percentile_80[0][0]),
                              (bbox_percentile_80[1][1] - bbox_percentile_80[1][0]),
                              (bbox_percentile_80[0][1] - bbox_percentile_80[0][0]),
                              linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((bbox_kmeans[1][0], bbox_kmeans[0][0]), (bbox_kmeans[1][1] - bbox_kmeans[1][0]),
                              (bbox_kmeans[0][1] - bbox_kmeans[0][0]),
                              linewidth=1, edgecolor='b', facecolor='none')
    current_axis.add_patch(rect1)
    current_axis.add_patch(rect2)
    plt.savefig(f"{path}/{idx}.png")
    plt.cla()


def get_2D_image(img):
    """
    turn 3D to 2D
    :param img: 3D MRI input
    :return:
    """
    return img[:, :, img.shape[2] // 2]


def run_crop(batch, img_folder, label_folder) -> int:
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

    cropped_img, cropped_label = crop_to_nonzero(img, label)
    cropped_img_file = nib.Nifti1Image(cropped_img, img_affine)
    nib.save(cropped_img_file, img_folder / Path(f"{filename}.gz"))
    cropped_label_file = nib.Nifti1Image(cropped_label, label_affine)
    nib.save(cropped_label_file, label_folder / Path(f"{filename}.gz"))
    return 1


def pre_transform() -> Compose:
    transform = Compose([
        Resample(1.0),
    ])
    return transform


if __name__ == "__main__":
    if not os.path.exists(cropped_img_folder):
        os.mkdir(cropped_img_folder)
    if not os.path.exists(cropped_label_folder):
        os.mkdir(cropped_label_folder)

    print(f"{ctime()}: starting ...")

    # for idx, mri in enumerate(get_path(datasets)):
    # if not COMPUTECANADA:
    # run_crop(idx, mri.img_path, mri.label_path, cropped_img_folder, cropped_label_folder)

    idx = 0

    subjects, visual_img_path_list, visual_label_path_list = get_subjects()

    transform = pre_transform()
    image_dataset = tio.ImagesDataset(subjects, transform=transform)
    loader = DataLoader(image_dataset,
                        batch_size=1)  # always one because using different label size

    for batch in tqdm(loader):
        idx += run_crop(batch, cropped_img_folder, cropped_label_folder)

    print(f"{ctime()}: ending ...")
    print(f"Totally get {idx} imgs!")
    # show_save_img_and_label(img_2D, label_2D, bbox_percentile_80, bbox_kmeans, "./rectangle_image", idx)
