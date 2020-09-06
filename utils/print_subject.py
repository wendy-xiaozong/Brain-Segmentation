import torchio
import torch
from torchio import DATA, PATH

from torch.utils.data import DataLoader
from data.get_subjects import get_subjects
from data.transform import get_val_transform, get_train_transforms


if __name__ == "__main__":
    subjects, visual_img_path_list, visual_label_path_list = get_subjects(use_cropped_resampled_data=False)
    # test_imageDataset = torchio.ImagesDataset(subjects[:1])
    cur_subject = subjects[0]

    # patches_validation_set = torchio.Queue(
    #     subjects_dataset=val_imageDataset,
    #     max_length=self.max_queue_length,
    #     samples_per_volume=self.samples_per_volume,
    #     sampler=torchio.sampler.UniformSampler(self.patch_size),
    #     num_workers=self.num_workers,
    #     shuffle_subjects=False,
    #     shuffle_patches=True,
    #     verbose=True,
    # )
    train_min, train_max, val_min, val_max = 0, 0, 0, 0
    for i in range(100):
        train_transform = get_train_transforms()
        train_subject = train_transform(cur_subject)
        val_transform = get_val_transform()
        val_subject = val_transform(cur_subject)

        train_max += train_subject['img'].data.max()
        train_min += train_subject['img'].data.min()
        val_max += val_subject['img'].data.max()
        val_min += val_subject['img'].data.min()

    # print(f"img: {subject['img'].data}")
    # print(f"type: {type(subject['img'].data)}")
    # print(f"label: {subject['label'].data}")

    print(f"train img range: [{train_min / 100}, {train_max / 100}]")
    print(f"val img range: [{val_min / 100}, {train_max / 100}]")

    # # the batch_size here only could be 1 because we only could handle one image to aggregate
    # test_loader = DataLoader(test_imageDataset, batch_size=1)
    #
    # for batch_idx, batch in enumerate(test_loader):
    #     inputs_path, targets_path = batch["img"][PATH], batch["label"][PATH]
    #     print(f"inputs path: {inputs_path[0]}")
    #     print(f"targets path: {targets_path[0]}")
    #     print(f"input type: {type(inputs_path[0])}")
