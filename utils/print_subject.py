import torchio
from torchio import DATA, PATH

from torch.utils.data import DataLoader
from data.get_subjects import get_processed_subjects


if __name__ == "__main__":
    subjects, visual_img_path_list, visual_label_path_list = get_processed_subjects()
    test_imageDataset = torchio.ImagesDataset(subjects[:2])

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

    # the batch_size here only could be 1 because we only could handle one image to aggregate
    test_loader = DataLoader(test_imageDataset, batch_size=1)

    for batch_idx, batch in enumerate(test_loader):
        inputs_path, targets_path = batch["img"][PATH], batch["label"][PATH]
        print(f"inputs path: {inputs_path[0]}")
        print(f"targets path: {targets_path[0]}")
        print(f"input type: {type(inputs_path[0])}")
