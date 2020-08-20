import os
import numpy as np
import shutil
from multiprocessing import Pool
from collections import OrderedDict
from sklearn.cluster import KMeans, MiniBatchKMeans
from data.const import COMPUTECANADA, DATA_ROOT, ADNI_DATASET_DIR_1, CC359_DATASET_DIR, NFBS_DATASET_DIR
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Union
from collections import OrderedDict
import matplotlib.patches as patches
import tqdm
import nibabel as nib
from time import ctime
from numpy import ndarray
from data.get_path import get_path
from matplotlib.pyplot import Axes, Figure
from skimage.transform import resize
from functools import reduce
import copy


class BrainSlices:
    def __init__(self, idx: int, img: ndarray, target_: ndarray):
        # lol mypy type inference really breaks down here...
        img_: ndarray = img
        targ_: ndarray = target_
        mids: ndarray = np.array(img_.shape) // 2
        quarts: ndarray = mids // 2  # slices at first quarter of the way through
        quarts3_4: ndarray = mids + quarts  # slices 3/4 of the way through
        self.mids = mids
        self.idx = idx
        self.quarts = quarts
        self.quarts3_4 = quarts3_4
        self.slice_positions = ["1/4", "1/2", "3/4"]
        self.shape = np.array(img_.shape)

        self.imgs = OrderedDict(
            [
                ("1/4", (img_[quarts[0], :, :], img_[:, quarts[1], :], img_[:, :, quarts[2]])),
                ("1/2", (img_[mids[0], :, :], img_[:, mids[1], :], img_[:, :, mids[2]])),
                ("3/4", (img_[quarts3_4[0], :, :], img_[:, quarts3_4[1], :], img_[:, :, quarts3_4[2]])),
            ]
        )
        self.targets = OrderedDict(
            [
                ("1/4", (targ_[quarts[0], :, :], targ_[:, quarts[1], :], targ_[:, :, quarts[2]])),
                ("1/2", (targ_[mids[0], :, :], targ_[:, mids[1], :], targ_[:, :, mids[2]])),
                ("3/4", (targ_[quarts3_4[0], :, :], targ_[:, quarts3_4[1], :], targ_[:, :, quarts3_4[2]])),
            ]
        )
        self.labels = {
            "1/4": [f"[{quarts[0]},:,:]", f"[:,{quarts[1]},:]", f"[:,:,{quarts[2]}]"],
            "1/2": [f"[{mids[0]},:,:]", f"[:,{mids[1]},:]", f"[:,:,{mids[2]}]"],
            "3/4": [f"[{quarts3_4[0]},:,:]", f"[:,{quarts3_4[1]},:]", f"[:,:,{quarts3_4[2]}]"],
        }

    def plot(self) -> Tuple[Figure, Axes]:
        nrows, ncols = 3, 1  # one row for each slice position
        all_trues, all_targets, all_preds = [], [], []
        for i in range(3):  # We want this first so middle images are middle
            for j, position in enumerate(self.slice_positions):
                img, target = self.imgs[position][i], self.targets[position][i]
                all_trues.append(img)
                all_targets.append(target)
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        true = np.concatenate(all_trues, axis=1)
        target = np.concatenate(all_targets, axis=1)

        # # Consistently apply colormap since images are standardized but still
        # # vary considerably in maximum and minimum values
        # true_args = dict(vmin=-3.0, vmax=8.0, cmap="gray")
        mask_args = dict(vmin=0.0, vmax=1.0, cmap="gray", alpha=0.5)

        axes[0].imshow(true)
        axes[0].imshow(target, **mask_args)
        axes[0].set_title("Actual Brain Tissue (probability)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        return fig, axes

    def visualize(self, outdir: Path = None) -> None:
        fig, axes = self.plot()
        fig.set_size_inches(w=10, h=6)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outdir / f"{self.idx}.png", dpi=200)
        plt.close()
        return


if __name__ == "__main__":
    if COMPUTECANADA:
        # DATA_ROOT = Path(str(os.environ.get("SLURM_TMPDIR"))).resolve()
        # DATA_ROOT = Path("/project/6005889/U-Net_MRI-Data")
        # cropped_img_folder = DATA_ROOT / "work" / "img"
        # cropped_label_folder = DATA_ROOT / "work" / "label"
        cropped_img_folder = DATA_ROOT / "img"
        cropped_label_folder = DATA_ROOT / "label"
    else:
        DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"
        img_path = DATA_ROOT / "all_different_size_img/img"
        label_path = DATA_ROOT / "all_different_size_img/label"
        cropped_img_folder = DATA_ROOT / "cropped" / "img"
        cropped_label_folder = DATA_ROOT / "cropped" / "label"

    if not os.path.exists(cropped_img_folder):
        os.mkdir(cropped_img_folder)
    if not os.path.exists(cropped_label_folder):
        os.mkdir(cropped_label_folder)

    img_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(cropped_img_folder)}/**/*.nii*", recursive=True))
    ])

    label_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(cropped_label_folder)}/**/*.nii.gz", recursive=True))
    ])

    print(f"{ctime()}: starting ...")
    # pool.map(_run_crop, arg_list[:16])

    # if COMPUTECANADA:
    #     datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    # else:
    #     datasets = [CC359_DATASET_DIR]

    # for idx, mri in enumerate(get_path(datasets)):
        # if not COMPUTECANADA:
        # run_crop(idx, mri.img_path, mri.label_path, cropped_img_folder, cropped_label_folder)

    idx = 0
    for img_path, label_path in zip(img_path_list, label_path_list):
        idx += 1
        img = nib.load(img_path)
        mask = nib.load(label_path)

        dims = list(img.shape)

        img_np = img.get_fdata()
        mask_np = mask.get_fdata()

        img_np = resize(img_np, output_shape=(128, 128, 128), mode='constant', anti_aliasing=True)
        mask_np = resize(mask_np, output_shape=(128, 128, 128), mode='constant', anti_aliasing=True)
        BrainSlices(idx, img_np, mask_np).visualize(Path("./plot"))
        print(f"{ctime()}: save plot {idx}!")

    # for mri in get_path(datasets):
    #     idx += 1
    #     run_crop(mri.img_path, mri.label_path, cropped_img_folder, cropped_label_folder)

    print(f"{ctime()}: ending ...")
    print(f"Totally plot {idx} imgs!")
    # show_save_img_and_label(img_2D, label_2D, bbox_percentile_80, bbox_kmeans, "./rectangle_image", idx)