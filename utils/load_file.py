import os
import nibabel as nib
import numpy as np
from pathlib import Path
from data.const import DATA_ROOT
import matplotlib.pyplot as plt


def get_2D_image(img):
    """
    turn 3D to 2D
    :param img: 3D MRI input
    :return:
    """
    return img[:, :, img.shape[2] // 2]


if __name__ == "__main__":
    img_folder = DATA_ROOT / "cropped" / "img"

    img_path = "/home/jq/PycharmProjects/Unet_seg138/Data/strange_MRI" \
               "/ADNI_021_S_0276_MR_MPR__GradWarp__N3__Scaled_Br_20061217133727654_S19508_I33469.nii"

    img = nib.load(img_path, mmap=False)
    data_np = img.get_data()

    if np.isnan(data_np).any():
        print("there is nan in original image!")
        data_np[np.isnan(data_np)] = 0.0

    if not np.isfinite(data_np).all():
        print("there is infinite number in the image")
        # find the max and min
        # max = np.nanmax(data_np[data_np != np.inf])
        # min = np.nanmin(data_np[data_np != -np.inf])
        # data_np[data_np == np.inf] = max
        # data_np[data_np == -np.inf] = min

    max = np.nanmax(data_np[data_np != np.inf])
    min = np.nanmin(data_np[data_np != -np.inf])
    print(f"before convert: [{min}, {max}]")

    data_np.astype(np.float)
    max = np.nanmax(data_np[data_np != np.inf])
    min = np.nanmin(data_np[data_np != -np.inf])
    print(f"after convert: [{min}, {max}]")

    if not np.isfinite(data_np).all():
        print("it still have infinite number in the image")

    # print(type(data_np))
    # plt.imshow(get_2D_image(data_np))
    # plt.savefig("tmp.png")

    # img_file = nib.Nifti1Image(data_np, img.affine, img.header)
    #
    # _, filename = os.path.split(img_path)
    # filename, _ = os.path.splitext(filename)
    # nib.save(img_file, img_folder / f"{filename}.nii")
    #
    # img = nib.load(img_folder / f"{filename}.nii", mmap=False)
    # data_np = img.get_data().astype(np.float)
    # print(type(data_np))
    # plt.imshow(get_2D_image(data_np))
    # plt.savefig("new_img.png")
