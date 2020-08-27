"""try to use the whole dataset to predict, and test the result
"""
from torchio import DATA
from pathlib import Path
from glob import glob
import torchio as tio
import nibabel as nib
import numpy as np
from time import ctime
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import os

from data.const import ADNI_DATASET_DIR_1, squeezed_img_folder, squeezed_label_folder
from data.transform import get_train_transforms
from data.get_path import get_path

# def _prepare_data(batch):
#     inputs, targets = batch["img"][DATA], batch["label"][DATA]
#     if torch.isnan(inputs).any():
#         print("there is nan in input data!")
#         inputs[inputs != inputs] = 0
#     # if torch.isnan(targets).any():
#     #     print("there is nan in targets data!")
#     #     targets[targets != targets] = 0
#     # making the label as binary, it is very strange because if the label is not binary
#     # the whole model cannot learn at all
#     # target_bin = torch.zeros(size=targets.size()).type_as(inputs)
#     # target_bin[targets > 0.5] = 1
#     return inputs, targets


def read_data(mri):
    img, label = nib.load(mri.img_path), nib.load(mri.label_path)
    data_np = img.get_data()
    seg_np = label.get_data().squeeze()

    # print(f"{data_np.dtype.type})

    return data_np, seg_np, img.affine, label.affine


if __name__ == "__main__":
    if not os.path.exists(squeezed_img_folder):
        os.mkdir(squeezed_img_folder)
    if not os.path.exists(squeezed_label_folder):
        os.mkdir(squeezed_label_folder)

    for mri in tqdm(get_path(ADNI_DATASET_DIR_1)):
        try:
            data_np, seg_np, img_affine, label_affine = read_data(mri)
        except OSError:
            print(f"{mri.img_path} this file is broken!")
            continue

        if data_np.shape != seg_np.shape:
            print(f"{mri.img_path}'s shape is not the same as it's label's shape!")
            continue

        # get the file name
        _, filename = os.path.split(mri.img_path)
        filename, _ = os.path.splitext(filename)

        img_file = nib.Nifti1Image(data_np, img_affine)
        nib.save(img_file, squeezed_img_folder / Path(f"{filename}.nii"))
        label_file = nib.Nifti1Image(seg_np, label_affine)
        nib.save(label_file, squeezed_label_folder / Path(f"{filename}.nii.gz"))

        # print(f"{ctime()}: Successfully save file {filename} file!")

        # subject = tio.Subject(t1=tio.Image(mri.img_path, type=tio.INTENSITY))

        # From NiftyNet model zoo
        # LI_LANDMARKS = "4.4408920985e-16 8.06305571158 15.5085721044 " \
        #                "18.7007018006 21.5032879029 26.1413278906 29.9862059045 33.8384058795 38.1891334787 " \
        #                "40.7217966068 44.0109152758 58.3906435207 100.0"
        # li_landmarks = np.array([float(n) for n in LI_LANDMARKS.split()])
        #
        # transforms = [
        #     tio.ToCanonical(),
        #     tio.Resample(1),
        #     # tio.HistogramStandardization(landmarks={'t1': li_landmarks}, masking_method=tio.ZNormalization.mean),
        #     tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        # ]
        # transform = tio.Compose(transforms)
        # preprocessed = transform(img_tensor)
        # break
