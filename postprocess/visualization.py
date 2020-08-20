from pathlib import Path
from typing import List, Tuple, Optional, Union, Any

import nibabel as nib
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import Axes, Figure
import matplotlib.pyplot as plt

# What is this used for?
from skimage.exposure import rescale_intensity


def plot_parameters(
        model: torch.nn.Module,
        parameters_name: str,
        title: Optional[str] = None,
        axis: Optional[plt.Axes] = None,
        kde: bool = True,
        kde_bandwidth: float = None,
) -> None:
    for name_, params in model.named_parameters():
        if name_ == parameters_name:
            tensor = params.data
            break
    else:
        raise ValueError(f'{parameters_name} not found in model')
    array = tensor.numpy().ravel()
    if axis is None:
        fig, axis = plt.subplots()
    if kde_bandwidth is None:
        sns.distplot(array, ax=axis, kde=kde)
    else:
        sns.kdeplot(array, ax=axis, bw=kde_bandwidth)
    if title is not None:
        axis.set_title(title)


def plot_all_parameters(
        model: torch.nn.Module,
        labelsize: int = 6,
        kde: bool = True,
        kde_bandwidth: float = None,
) -> None:
    fig, axes = plt.subplots(3, 7, figsize=(11, 5))
    axes = list(reversed(axes.ravel()))
    for parameters_name, params in model.named_parameters():
        if len(params.data.shape) < 2:
            continue
        axis = axes.pop()
        plot_parameters(
            model,
            parameters_name,
            axis=axis,
            kde=kde,
            kde_bandwidth=kde_bandwidth,
        )
        axis.xaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()


def to_rgb(array: np.ndarray) -> np.ndarray:
    if array.shape[-1] == 3:  # assume it's already RGB
        return array
    array = array.astype(float)
    array -= array.min()
    array /= array.max()
    array *= 255
    array = array.astype(np.uint8)
    rgb = np.stack(3 * [array], axis=-1)
    return rgb


def rescale_array(
        array: np.ndarray,
        cutoff: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    percentiles = tuple(np.percentile(array, cutoff))
    array = rescale_intensity(array, in_range=percentiles)
    return array


def add_intersections(
        slices: Tuple[np.ndarray, np.ndarray, np.ndarray],
        i: int,
        j: int,
        k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Colors from 3D Slicer
    """
    sag, cor, axi = slices
    red = 255, 131, 114
    green = 143, 229, 97
    yellow = 255, 237, 135
    sag[j, :] = green
    sag[:, k] = red
    cor[i, :] = yellow
    cor[:, k] = red
    axi[i, :] = yellow
    axi[:, j] = green
    return sag, cor, axi


def plot_original_volume(
        input_img: np.ndarray,
        target_img: np.ndarray,
        predict_img: np.ndarray,
        colors_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        idx_sag: Optional[int] = None,
        idx_cor: Optional[int] = None,
        idx_axi: Optional[int] = None,
        return_figure: bool = False,
) -> Optional[plt.Figure]:
    """
    Expects an isotropic-spacing volume in RAS orientation
    """



# def plot_volume_interactive(array: np.ndarray, **kwargs) -> None:  # it only use this
#     def get_widget(size, description):
#         widget = widgets.IntSlider(
#             min=0,
#             max=size - 1,
#             step=1,
#             value=size // 2,
#             continuous_update=False,
#             description=description,
#         )
#         return widget
#
#     shape = array.shape[:3]
#     names = 'Sagittal L-R', 'Coronal P-A', 'Axial I-S'
#     widget_sag, widget_cor, widget_axi = [
#         get_widget(s, n) for (s, n) in zip(shape, names)]
#     ui = widgets.HBox([widget_sag, widget_cor, widget_axi])
#     args_dict = {
#         'array': fixed(array),
#         'idx_sag': widget_sag,
#         'idx_cor': widget_cor,
#         'idx_axi': widget_axi,
#         'return_figure': fixed(True),
#     }
#     kwargs = {key: fixed(value) for (key, value) in kwargs.items()}
#     args_dict.update(kwargs)
#     out = widgets.interactive_output(plot_volume, args_dict)
#     display(ui, out)


def plot_histogram(
        array: np.ndarray,
        kde: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        add_labels: bool = False,
) -> None:
    sns.distplot(array.ravel(), kde=kde)
    if ylim is not None:
        plt.ylim(ylim)
    if add_labels:
        plt.xlabel('Intensity')
        plt.ylabel('Number of voxels')


if __name__ == "__main__":
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "Data/all_different_size_img"
    IMG = DATA_ROOT / "img"
    LABEL = DATA_ROOT / "label"

    img_path = IMG / "ADNI_123_S_4127_MR_MT1__GradWarp__N3m_Br_20120906172542553_S159610_I331818.nii"
    label_path = LABEL / "ADNI_123_S_4127_MR_MT1__GradWarp__N3m_Br_20120906172542553_S159610_I331818.nii.gz"

    img, label = nib.load(img_path), nib.load(label_path)
    data_np = img.get_data().astype(np.uint8)
    seg_npy = label.get_data().squeeze().astype(np.float)

    fig = plot_volume(
        data_np,
        enhance=False,
        colors_path='GIFNiftyNet.ctbl',
        return_figure=True,
    )

    fig.savefig("./img.jpg")
