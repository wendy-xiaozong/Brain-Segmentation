import random
from data.get_subjects import get_processed_subjects
import torchio



if __name__ == "__main__":
    subjects, visual_img_path_list, visual_label_path_list = get_processed_subjects(
        whether_use_cropped_and_resample_img=True
    )
    random.seed(42)
    random.shuffle(subjects)  # shuffle it to pick the val set
    num_subjects = len(subjects)
    num_training_subjects = int(num_subjects * 0.97)  # （5074+359+21） * 0.9 used for training
    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    print("validation:")
    for id, validation_subject in enumerate(validation_subjects):
        print(f"    {id}| shape: {validation_subject.img.numpy().squeeze().shape}")

    print("visualidation:")
    for id, (img_path, label_path) in enumerate(zip(visual_img_path_list, visual_label_path_list)):
        subject = torchio.Subject(
            img=torchio.Image(path=img_path, type=torchio.INTENSITY)
        )
        print(f"    {id}| shape: {subject.img.numpy().squeeze().shape}")
        