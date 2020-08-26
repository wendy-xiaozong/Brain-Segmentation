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

    img_path = "/Data/ADNI/005_S_2390/MT1__GradWarp__N3m/2011-06-27_09_38_47.0/S112699/ADNI_005_S_2390_MR_MT1__GradWarp__N3m_Br_20110701094138392_S112699_I242887.nii"
    label_path = "/Data/label/MALPEM-ADNI_005_S_2390_MR_MT1__GradWarp__N3m_Br_20110701094138392_S112699_I242887.nii.gz"

    img = nib.load(img_path, mmap=False)
    data_np = img.get_data().astype(np.float)
    print(type(data_np))
    plt.imshow(get_2D_image(data_np))
    plt.savefig("tmp.png")

    img_file = nib.Nifti1Image(data_np, img.affine, img.header)

    _, filename = os.path.split(img_path)
    filename, _ = os.path.splitext(filename)
    nib.save(img_file, img_folder / f"{filename}.nii")

    img = nib.load(img_folder / f"{filename}.nii", mmap=False)
    data_np = img.get_data().astype(np.float)
    print(type(data_np))
    plt.imshow(get_2D_image(data_np))
    plt.savefig("new_img.png")
