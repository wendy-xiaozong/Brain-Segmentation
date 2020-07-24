"""create a MRI class for every instance, to load every instance's image and label path"""
from .const import *
import os
import re
import random
import nibabel as nib


class MRI:
    def __init__(self, dataset, file_name):
        """
        :param dataset: dataset name
        :param file_name: every instance
        """
        self.dataset = dataset
        self.file_name = file_name
        self.img_path = ""
        self.label_path = ""
        self.val_data = random.random()
        self.flag = True  # in case some file are not NIFTI file
        self.get_path()

    def get_path(self):
        """
        get NTFTi file path with orignial MRI and label path
        :return:
        """
        self.img_path = self.file_name
        label_file_name = str("MALPEM-" + self.file_name.name + '.gz')
        self.label_path = os.path.join(ADNI_LABEL, label_file_name)
        if not (os.path.exists(self.img_path) and os.path.exists(self.label_path)):
            self.flag = False

    def show_image_shape(self):
        """
        print the shape of the MRI
        :return: None
        """
        img = []
        seg = []
        try:
            # in case some times found the file isnt exist like ".xxx" file
            img = nib.load(self.img_path).get_data()
        except OSError as e:
            print("not such img file:", self.img_path)
        try:
            seg = nib.load(self.label_path).get_data().squeeze()
        except OSError as e:
            print("not such label file:", self.label_path)
        return img, seg
