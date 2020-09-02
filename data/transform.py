from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    RandomBlur,
    RandomSpike,
    RandomGhosting,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Pad,
    Compose,
)
from .custom_trans_class import ToSqueeze


def get_train_transforms() -> Compose:
    training_transform = Compose([
        ToCanonical(),
        Resample(1),  # this might need to change
        # Do I really need this? if I use this, I would have `FloatingPointError: underflow encountered in true_divide`
        # I don't know how to deal with it right now
        # It seems like when doing
        #     array /= array.max()  # [0, 1]
        # a very small number come out, cause it
        # very strange, because I never run into it before?
        # nnUnet do not use it in MRI, they only use it in CT
        # RescaleIntensity(
        #     out_min_max=(0, 1),
        #     percentiles=(0.5, 99.5)  # what this used for?
        # ),
        # so that there are no negative values for RandomMotion
        # this might not work if I don't use the RescaleIntensity above
        # might be add this:
        # HistogramStandardization({'mri': landmarks}),
        ZNormalization(masking_method=ZNormalization.mean),  # Subtract mean and divide by standard deviation.
        RandomMotion(
            degrees=10,
            translation=10,
            num_transforms=2,
            p=0.2,
            # seed=seed,
        ),
        RandomBlur(
            std=(0, 4),
            p=0.2,
            # seed=seed,
        ),  # ??
        RandomSpike(
            num_spikes=1,
            # Ratio r between the spike intensity and the maximum of the spectrum.
            # Larger values generate more distorted images.
            intensity=(1, 3),
            p=0.2,
            # seed=seed,
        ),
        RandomBiasField(
            coefficients=0.5,
            order=3,
            p=0.1,
            # seed=seed,
        ),
        RandomFlip(
            axes=(0, 1, 2),
            p=0.5,
            # seed=seed,
        ),  # this probability might need to tune
        RandomGhosting(
            num_ghosts=(2, 10),
            intensity=(0.5, 1),
            p=0.01,
            # seed=seed,
        ),
        OneOf({
            RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                translation=5,
                # seed=seed
            ): 0.8,
            RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.5,
                # seed=seed,
            ): 0.2,
        }),
        RandomNoise(
            mean=0,
            std=(0, 0.25),
            p=0.25,
        ),
    ])

    return training_transform


def get_val_transform() -> Compose:
    validation_transform = Compose([
        ToCanonical(),
        Resample(1),  # this might need to change
        # RescaleIntensity((0, 1)),
        ZNormalization(masking_method=ZNormalization.mean),
    ])
    return validation_transform


def get_test_transform() -> Compose:  # do not resize in there
    validation_transform = Compose([
        ToCanonical(),
        Resample(1),
        # CropOrPad(64),
        # RescaleIntensity((0, 1)),
        # ToResize_only_image(),
        ZNormalization(masking_method=ZNormalization.mean),
    ])
    return validation_transform
