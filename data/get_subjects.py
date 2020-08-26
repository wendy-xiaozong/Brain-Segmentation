"""
using TorchIO to do 3D MRI augmentation, need pytorch as framework
"""
import random
from .const import *
import torchio as tio
from time import ctime
from .get_path import get_path, get_1069_path
from .const import cropped_img_folder, cropped_label_folder, DATA_ROOT
from glob import glob
import pandas as pd


def get_original_subjects():
    """
    get data from the path and do augmentation on it, and return a DataLoader
    :return: list of subjects
    """

    if COMPUTECANADA:
        datasets = [ADNI_DATASET_DIR_1]
    else:
        datasets = [ADNI_DATASET_DIR_1]

    subjects = [
        tio.Subject(
                img=tio.Image(path=mri.img_path, type=tio.INTENSITY),
                label=tio.Image(path=mri.label_path, type=tio.LABEL),
                # store the dataset name to help plot the image later
                # dataset=mri.dataset
            ) for mri in get_path(datasets)
    ]

    visual_img_path_list = []
    visual_label_path_list = []

    for mri in get_1069_path(datasets):
        visual_img_path_list.append(mri.img_path)
        visual_label_path_list.append(mri.label_path)

    # using in the cropping folder
    # img_path_list = sorted([
    #     Path(f) for f in sorted(glob(f"{str(CROPPED_IMG)}/**/*.nii*", recursive=True))
    # ])
    # label_path_list = sorted([
    #     Path(f) for f in sorted(glob(f"{str(CROPPED_LABEL)}/**/*.nii.gz", recursive=True))
    # ])
    #
    # subjects = [
    #     tio.Subject(
    #             img=tio.Image(path=img_path, type=tio.INTENSITY),
    #             label=tio.Image(path=label_path, type=tio.LABEL),
    #             # store the dataset name to help plot the image later
    #             # dataset=mri.dataset
    #         ) for img_path, label_path in zip(img_path_list, label_path_list)
    # ]

    print(f"{ctime()}: getting number of subjects {len(subjects)}")
    print(f"{ctime()}: getting number of path for visualizationg {len(visual_img_path_list)}")
    return subjects, visual_img_path_list, visual_label_path_list


def get_subjects(
        use_cropped_data: True
):
    if use_cropped_data:
        # using in the cropping folder
        img_path_list = sorted([
            Path(f) for f in sorted(glob(f"{str(cropped_img_folder)}/**/*.nii*", recursive=True))
        ])
        label_path_list = sorted([
            Path(f) for f in sorted(glob(f"{str(cropped_label_folder)}/**/*.nii.gz", recursive=True))
        ])
    else:
        img_path_list = sorted([
            Path(f) for f in sorted(glob(f"{str(squeezed_img_folder)}/**/*.nii*", recursive=True))
        ])
        label_path_list = sorted([
            Path(f) for f in sorted(glob(f"{str(squeezed_label_folder)}/**/*.nii.gz", recursive=True))
        ])

    subjects = [
        tio.Subject(
            img=tio.Image(path=img_path, type=tio.INTENSITY),
            label=tio.Image(path=label_path, type=tio.LABEL),
            # store the dataset name to help plot the image later
            # dataset=mri.dataset
        ) for img_path, label_path in zip(img_path_list, label_path_list)
    ]

    fine_tune_set_file = Path(__file__).resolve().parent.parent.parent / "ADNI_MALPEM_baseline_1069.csv"
    file_df = pd.read_csv(fine_tune_set_file, sep=',')
    images_baseline_set = set(file_df['filename'])
    random.seed(42)
    images_baseline_set = random.sample(images_baseline_set, 150)

    visual_img_path_list = []
    visual_label_path_list = []

    # used for visualization
    for img_path in img_path_list:
        img_name = img_path.name
        if img_name in images_baseline_set:
            visual_img_path_list.append(img_path)
    for label_path in label_path_list:
        label_name = label_path.name
        if label_name in images_baseline_set:
            visual_label_path_list.append(label_path)

    print(f"{ctime()}: getting number of subjects {len(subjects)}")
    print(f"{ctime()}: getting number of path for visualizationg {len(visual_img_path_list)}")
    return subjects, visual_img_path_list, visual_label_path_list