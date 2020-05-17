import os
import pickle
from random import shuffle

import config
import SimpleITK as sitk
import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS
path = "../Data/training"
data_list = ["1", "4", "5", "7", "14", "070", "148"]


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
        print("get shape:", self.shape)
        self.spacing = self.flair_img.GetSpacing()  #
        print("get spacing:", self.spacing)
        self.origin = self.flair_img.GetOrigin()
        print("get origin:", self.origin)
        self.direction = self.flair_img.GetDirection()
        print("get direction:", self.direction)

    def __str__(self):
        return "Subject: {}\nData path: {}\n".format(self.name,
                                                     self.path)


def get_objects(files, path=path):
    print("path:", path)
    print("data_dir", FLAGS.data_dir)
    subjects_dict = {"train": [], "val": []}
    # files files = {"train": data_list[:train_subjects], "val": data_list[train_subjects:]}
    for key in files.keys():
        n_files = len(files[key])  # firstly 5
        print("\n{} {} data {}".format("*" * 20, key, "*" * 20))
        for n in range(n_files):
            subject = files[key][n]
            subjects_dict[key].append(Subjects(subject, path))
            print("Subject: " + subject)
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


def get_dataset():
    # get a dict to save where we run
    dataset_files = get_files(checkpoint=FLAGS.files_checkpoint,  # $work_dir/train/run_$RUN/checkpoints/run_${
                              # RUN}_data.p
                              train_subjects=FLAGS.train_subjects)  # 5
    dataset = get_objects(dataset_files)
    # add_extra_dims(dataset)
    return dataset


if __name__ == '__main__':
    get_dataset()