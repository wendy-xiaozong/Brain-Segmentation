from .const import ADNI_DATASET_DIR_1, ADNI_DATASET_DIR_2, ADNI_LABEL
from .MRI import MRI
import os
from pathlib import Path
import pandas as pd
import random
import re


def get_path(dataset):
    brain_label = ADNI_LABEL
    originals = list(Path(ADNI_DATASET_DIR_1).glob("**/*.nii"))
    originals.extend(list(Path(ADNI_DATASET_DIR_2).glob("**/*.nii")))
    # print(f"len originals: {len(originals)}")

    print(f"get {len(originals)} of imgs")
    regex = re.compile(r"MALPEM-ADNI_(.*?).nii.gz")
    brain_label_set = set(label for label in os.listdir(brain_label) if regex.match(label))

    for original in originals:
        cur = Path(original)

        # print("stem:", cur.stem)  # without suffix
        # print("name:", cur.name)  # with suffix
        label_file = "MALPEM-" + cur.name + ".gz"
        if label_file in brain_label_set:
            mri = MRI(dataset, original)
            if mri.flag:
                yield mri


def get_1069_path(dataset):
    brain_label = ADNI_LABEL
    originals = list(Path(ADNI_DATASET_DIR_1).glob("**/*.nii"))
    originals.extend(list(Path(ADNI_DATASET_DIR_2).glob("**/*.nii")))
    # print(f"len originals: {len(originals)}")

    fine_tune_set_file = Path(__file__).resolve().parent.parent.parent / "ADNI_MALPEM_baseline_1069.csv"
    file_df = pd.read_csv(fine_tune_set_file, sep=',')
    images_baseline_set = set(file_df['filename'])

    regex = re.compile(r"MALPEM-ADNI_(.*?).nii.gz")
    brain_label_set = set(label for label in os.listdir(brain_label) if regex.match(label))

    random.seed(42)
    images_baseline_set = random.sample(images_baseline_set, 150)

    for original in originals:
        cur = Path(original)

        # print("stem:", cur.stem)  # without suffix
        # print("name:", cur.name)  # with suffix
        label_file = "MALPEM-" + cur.name + ".gz"
        baseline_set_name = cur.name + ".gz"
        if label_file in brain_label_set and baseline_set_name in images_baseline_set:
            mri = MRI(dataset, original)
            if mri.flag:
                yield mri