import torchio
import torch
from torchio import DATA, PATH

from torch.utils.data import DataLoader
from data.get_subjects import get_subjects
from data.transform import get_val_transform


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
    # val_transform = get_val_transform()
    # subject = val_transform(cur_subject)
    # print(f"img: {subject['img'].data}")
    # print(f"type: {type(subject['img'].data)}")
    # print(f"label: {subject['label'].data}")

    a = torch.tensor([])
    b = torch.tensor([1.0])
    a = torch.cat((a, b), dim=0)
    print(f"a: {a}")

    # # the batch_size here only could be 1 because we only could handle one image to aggregate
    # test_loader = DataLoader(test_imageDataset, batch_size=1)
    #
    # for batch_idx, batch in enumerate(test_loader):
    #     inputs_path, targets_path = batch["img"][PATH], batch["label"][PATH]
    #     print(f"inputs path: {inputs_path[0]}")
    #     print(f"targets path: {targets_path[0]}")
    #     print(f"input type: {type(inputs_path[0])}")
