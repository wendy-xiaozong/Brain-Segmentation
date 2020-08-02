"""Some code is from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes, Figure
from matplotlib.text import Text
from numpy import ndarray
import numpy as np
from tqdm.auto import tqdm, trange
import os
import torch as t

from collections import OrderedDict
from numpy import ndarray
from matplotlib.pyplot import Axes, Figure
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Optional
from pytorch_lightning.core.lightning import LightningModule
import pandas as pd
import matplotlib.gridspec as gridspec

import sys
sys.path.append('../data/')
from data.const import colors_path

"""
For TensorBoard logging usage, see:
https://www.tensorflow.org/api_docs/python/tf/summary

For Lightning documentation / examples, see:
https://pytorch-lightning.readthedocs.io/en/latest/experiment_logging.html#tensorboard

NOTE: The Lightning documentation here is not obvious to newcomers. However,
`self.logger` returns the Torch TensorBoardLogger object (generally quite
useless) and `self.logger.experiment` returns the actual TensorFlow
SummaryWriter object (e.g. with all the methods you actually care about)

For the Lightning methods to access the TensorBoard .summary() features, see
https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.html#pytorch_lightning.loggers.TensorBoardLogger

**kwargs for SummaryWriter constructor defined at
https://www.tensorflow.org/api_docs/python/tf/summary/create_file_writer
^^ these args look largely like things we don't care about ^^
"""


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> [ndarray]:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue. """
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    if len(img.shape) == 3:
        return [scaled] * 3
    raise ValueError("Only accepts 1-channel or 3-channel images")


def get_logger(logdir: Path) -> TensorBoardLogger:
    return TensorBoardLogger(str(logdir), name="unet")


class ColorTable:
    def __init__(self, colors_path: Union[str, Path]):
        self.df = self.read_color_table(colors_path)

    @staticmethod
    def read_color_table(colors_path: Union[str, Path]):
        df = pd.read_csv(
            colors_path,
            sep=' ',
            header=None,
            names=[
                'Label',
                'Name',
                'R',
                'G',
                'B',
                'A',
            ],
            index_col='Label'
        )
        return df

    def get_color(self, label: int) -> Tuple[int, int, int]:
        """
        There must be nicer ways of doing this
        """
        try:
            rgb = (
                self.df.loc[label].R,
                self.df.loc[label].G,
                self.df.loc[label].B,
            )
        except KeyError:
            rgb = 0, 0, 0
        return rgb

    def colorize(self, label_map: np.ndarray) -> np.ndarray:
        rgb = np.stack(3 * [label_map], axis=-1)
        for label in np.unique(label_map):
            mask = label_map == label
            rgb[mask] = self.get_color(label)
        return rgb


def turn(array_2d: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(array_2d))


# https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
class BrainSlices:
    def __init__(self, lightning: LightningModule,
                 img: Tensor,
                 target_: Tensor,
                 prediction: Tensor,
                 colors_path: Optional[Union[str, Path]] = None):
        # lol mypy type inference really breaks down here...
        self.lightning = lightning
        self.input_img: ndarray = img.cpu().detach().numpy().squeeze()
        # the float value need to cast to np.unit8, for ColorTable and plot
        self.target_img: ndarray = target_.numpy().squeeze().astype(np.uint8)
        self.predict_img: ndarray = prediction.cpu().detach().numpy().squeeze().astype(np.uint8)

        si, sj, sk = self.input_img.shape[:3]
        i = si // 2
        j = sj // 2
        k = sk // 2
        self.slices = [
            self.get_slice(self.input_img, i, j, k),
            self.get_slice(self.target_img, i, j, k),
            self.get_slice(self.predict_img, i, j, k)
        ]

        if colors_path is not None:
            color_table = ColorTable(colors_path)
            self.slices[1] = [color_table.colorize(s) for s in self.slices[1]]
            self.slices[2] = [color_table.colorize(s) for s in self.slices[2]]

        self.title = ["Actual Brain Tissue",
                      "Actual Brain Parcellation",
                      "Predicted Brain Parcellation"]
        self.shape = np.array(self.input_img.shape)

        # Those use for mp4
        self.masks = [np.ones([*self.input_img.shape], dtype=int), self.target_img, self.predict_img]
        self.mask_video_names = ["Actual Brain Tissue",
                                 "Actual Brain Parcellation",
                                 "Predicted Brain Parcellation"]
        self.scale_imgs = make_imgs(self.input_img)
        self.scale_imgs = np.where(self.masks, self.scale_imgs, 0)
        # print(f"masks shape: {self.masks.shape}")
        # print(f"scale_imgs shape: {self.scale_imgs.shape}")

    def get_slice(
            self,
            input: np.ndarray,
            i: int,
            j: int,
            k: int
    ):
        return [
            input[i, ...],
            input[:, j, ...],
            input[:, :, k, ...],
        ]

    def plot(self) -> Figure:
        nrows, ncols = 3, 3  # one row for each slice position

        fig = plt.figure(figsize=(15, 25))
        # need to change here
        # 160 is a random number
        gs = gridspec.GridSpec(nrows, ncols, width_ratios=[self.shape[1] / 160, 1, 1])

        for i in range(0, 3):
            ax1 = plt.subplot(gs[i*3])
            ax2 = plt.subplot(gs[i*3 + 1])
            ax3 = plt.subplot(gs[i*3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes, self.title[i], i)

        plt.tight_layout()
        return fig

    def plot_row(
            self,
            slices: List,
            axes: Tuple[Any, Any, Any],
            title: str,
            row_num: int,
    ) -> None:
        for (slice_, axis) in zip(slices, axes):
            if row_num == 0:
                axis.imshow(turn(slice_), cmap="bone", alpha=0.8)
            else:
                axis.imshow(turn(slice_))
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])
            if title is not None:
                plt.gcf().suptitle(title)

    def log(self, fig: Figure, dice_score: float) -> None:
        logger = self.lightning.logger
        summary = f"Epoch {self.lightning.current_epoch + 1} dice_score {dice_score}"
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)

    # code is borrowed from: https://github.com/DM-Berger/autocrop/blob/master/autocrop/visualize.py#L125
    def animate_masks(
            self,
            dpi: int = 100,
            n_frames: int = 128,
            fig_title: str = None,
            outfile: Path = None,
    ) -> None:
        def get_slice(img: ndarray, ratio: float) -> ndarray:
            """Returns eig_img, raw_img"""

            if ratio < 0 or ratio > 1:
                raise ValueError("Invalid slice position")
            if len(img.shape) == 3:
                x_max, y_max, z_max = np.array(img.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(img.shape) * ratio), dtype=int)
            elif len(img.shape) == 4:
                x_max, y_max, z_max, _ = np.array(img.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(img.shape[:-1]) * ratio), dtype=int)
            x = int(10 + ratio * (x_max - 20))  # make x go from 10:-10 of x_max
            y = int(10 + ratio * (y_max - 20))  # make x go from 10:-10 of x_max
            x = x - 1 if x == x_max else x
            y = y - 1 if y == y_max else y
            z = z - 1 if z == z_max else z
            img_np = np.concatenate([img[x, :, :], img[:, y, :], img[:, :, z]], axis=1)
            return img_np

        def init_frame(img: ndarray, ratio: float, fig: Figure, ax: Axes, title) -> Tuple[
            AxesImage, Colorbar, Text]:
            image_slice = get_slice(img, ratio=ratio)
            # the bigger alpha, the image would become more black
            true_args = dict(vmin=0, vmax=255, cmap="bone", alpha=0.8)

            im = ax.imshow(image_slice, animated=True, **true_args)
            # im = ax.imshow(image_slice, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            title = ax.set_title(title)
            cb = fig.colorbar(im, ax=ax)
            return im, cb, title

        def update_axis(img: ndarray, ratio: float, im: AxesImage) -> AxesImage:
            image_slice = get_slice(img, ratio=ratio)
            # mask_slice = get_slice(mask, ratio=ratio)

            # vn, vm = get_vranges()
            im.set_data(image_slice)
            # im.set_data(mask_slice)
            # im.set_clim(vn, vm)
            # we don't have to update cb, it is linked
            return im

        # owe a lot to below for animating the colorbars
        # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
        def init() -> Tuple[Figure, Axes, List[AxesImage], List[Colorbar]]:
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False)  # 3

            ims: List[AxesImage] = []
            cbs: List[Colorbar] = []

            for ax, img, mask, title in zip(axes.flat, self.scale_imgs, self.masks, self.mask_video_names):
                im, cb, title = init_frame(img=img, ratio=0.0, fig=fig, ax=ax, title=title)
                ims.append(im)
                cbs.append(cb)

            if fig_title is not None:
                fig.suptitle(fig_title)
            fig.tight_layout(h_pad=0)
            fig.set_size_inches(w=12, h=10)  # The width of the entire image displayed
            fig.subplots_adjust(hspace=0.2, wspace=0.0)
            return fig, axes, ims, cbs

        N_FRAMES = n_frames
        ratios = np.linspace(0, 1, num=N_FRAMES)

        fig, axes, ims, cbs = init()

        # awkward, but we need this defined after to close over the above variables
        def animate(f: int) -> Any:
            ratio = ratios[f]
            updated = []
            for im, img, mask in zip(ims, self.scale_imgs, self.masks):
                updated.append(update_axis(img=img, ratio=ratio, im=im))
            return updated

        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            frames=N_FRAMES,
            blit=False,
            interval=24000 / N_FRAMES,
            repeat_delay=100 if outfile is None else None,
        )

        if outfile is None:
            plt.show()
        else:
            pbar = tqdm(total=100, position=1, desc='mp4')

            def prog_logger(current_frame: int, total_frames: int = N_FRAMES) -> Any:
                if (current_frame % (total_frames // 10)) == 0 and (current_frame != 0):
                    pbar.update(10)
                # tqdm.write("Done task %i" % (100 * current_frame / total_frames))
                #     print("Saving... {:2.1f}%".format(100 * current_frame / total_frames))

            # writervideo = animation.FFMpegWriter(fps=60)
            ani.save(outfile, codec="h264", dpi=dpi, progress_callback=prog_logger)
            # ani.save(outfile, progress_callback=prog_logger, writer=writervideo)
            pbar.close()


def log_weights(module: LightningModule) -> None:
    for name, param in module.named_parameters():
        module.logger.experiment.add_histogram(name, param, global_step=module.global_step)


"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""


def log_all_info(module: LightningModule, img: Tensor, target: Tensor, preb: Tensor, dice_score: float) -> None:
    """Helper for decluttering training loop. Just performs all logging functions."""
    brainSlice = BrainSlices(module, img, target, preb, colors_path=colors_path)
    fig = brainSlice.plot()

    brainSlice.log(fig, dice_score)

    # mp4_path = Path(__file__).resolve().parent.parent / "mp4"
    # if not os.path.exists(mp4_path):
    #     os.mkdir(mp4_path)
    #
    # brainSlice.animate_masks(fig_title=f"epoch: {module.current_epoch}, batch: {batch_idx}, dice_score: {dice_score}",
    #                          outfile=mp4_path / Path(
    #                              f"epoch={module.current_epoch}_batch={batch_idx}_dice_score={dice_score}.mp4"))
    # log_weights(module)
