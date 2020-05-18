import os
import pickle
from random import shuffle

import config
import SimpleITK as sitk
import tensorflow as tf
import numpy as np

FLAGS = tf.compat.v1.flags.FLAGS
path = "../Data/training"
data_list = ["1", "4", "5", "7", "14", "070", "148"]  # "1", "4", "5", "7", "14" train,  "070", "148" validation


class Subjects(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path
        # open files
        self.flair_img = sitk.ReadImage(os.path.join(self.path,
                                                     self.name,
                                                     "pre/FLAIR.nii.gz"))
        self.t1_img = sitk.ReadImage(os.path.join(self.path,
                                                  self.name,
                                                  "pre/reg_T1.nii.gz"))
        self.ir_img = sitk.ReadImage(os.path.join(self.path,
                                                  self.name,
                                                  "pre/reg_IR.nii.gz"))
        self.label_img = sitk.ReadImage(os.path.join(self.path,
                                                     self.name,
                                                     "segm.nii.gz"))
        # get data array
        self.flair_array = sitk.GetArrayFromImage(self.flair_img)
        self.t1_array = sitk.GetArrayFromImage(self.t1_img)
        self.ir_array = sitk.GetArrayFromImage(self.ir_img)
        self.label_array = sitk.GetArrayFromImage(self.label_img)
        self.shape = self.flair_array.shape

    def __str__(self):
        return "Subject: {}\nData path: {}\n".format(self.name,
                                                     self.path)

    def add_dims_3d(self):
        self.label_shape = tuple([1] + list(self.shape) + [11])  # (1, 48, 240, 240, 11)
        self.shape = tuple([1] + list(self.shape) + [1])  # (1, 48, 240, 240, 1)
        self.flair_array = np.reshape(self.flair_array, self.shape)  # before (48, 240, 240) after (1, 48, 240, 240, 1)
        self.t1_array = np.reshape(self.t1_array, self.shape)
        self.ir_array = np.reshape(self.ir_array, self.shape)
        self.label_array = one_hot_encode(self.label_array, self.label_shape)


def get_objects(files, path=path):
    subjects_dict = {"train": [], "val": []}
    # files files = {"train": data_list[:train_subjects], "val": data_list[train_subjects:]}
    for key in files.keys():
        n_files = len(files[key])  # firstly 5
        print("\n{} {} data {}".format("*" * 20, key, "*" * 20))
        for n in range(n_files):
            subject = files[key][n]
            subjects_dict[key].append(Subjects(subject, FLAGS.data_dir))
            print("get data! Subject: " + subject)
    print()
    return subjects_dict


def get_files(checkpoint="train/run_1/checkpoints/run_1",
              train_subjects=5,
              data_list=data_list,
              shuffle_files=True):
    try:
        files = load_files_order(checkpoint)
    except FileNotFoundError:
        if shuffle_files:
            shuffle(data_list)
        files = {"train": data_list[:train_subjects], "val": data_list[train_subjects:]}
        save_files_order(files, checkpoint)
    return files


def save_files_order(files, checkpoint):
    with open(checkpoint, 'wb') as handle:
        pickle.dump(files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    print('File ' + checkpoint + ' saved\n')


def load_files_order(checkpoint):
    with open(checkpoint, 'rb') as handle:
        files = pickle.load(handle)
        handle.close()
    print('File ' + checkpoint + ' loaded\n')
    return files


def add_extra_dims(subjects_dict, dims=3):
    for key in subjects_dict.keys():
        print("key:", key)
        n_files = len(subjects_dict[key])
        for n in range(n_files):
            subject = subjects_dict[key][n]
            if dims == 3:
                subject.add_dims_3d()


def one_hot_encode(array, shape):
    array += 1
    channels = shape[-1]  # (1, 48, 240, 240, 11)
    array = np.repeat(array, channels)  # channels = 11 repeat all the number 11 times to make one hot
    array = np.reshape(array, shape)  # before (30412800,) after  (1, 48, 240, 240, 11)
    array = array / np.arange(1, channels + 1)
    array[array > 1] = 0
    array[array < 1] = 0  # only one number = 1
    return array


def get_dataset():
    # get a dict to save where we run
    dataset_files = get_files(checkpoint=FLAGS.files_checkpoint,  # $work_dir/train/run_$RUN/checkpoints/run_${
                              # RUN}_data.p
                              train_subjects=FLAGS.train_subjects)  # 5
    dataset = get_objects(dataset_files)
    add_extra_dims(dataset)
    return dataset


if __name__ == '__main__':
    get_dataset()
