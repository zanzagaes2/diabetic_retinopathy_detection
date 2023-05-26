from utilities.transformations import *


def train_transform(image, label):
    return (light_augmentation_and_normalize(
        height=512,
        width=512,
        mean=NormalizingConstants.train_mean,
        std=NormalizingConstants.train_std
    )(image=image)["image"], label)


def rotate(image, label):
    return (rotate_and_normalize(
        height=512,
        width=512,
        mean=NormalizingConstants.train_mean,
        std=NormalizingConstants.train_std
    )(image=image)["image"], label)


def normalize_image(image, label):
    return (normalize(
        height=512,
        width=512,
        mean=NormalizingConstants.train_mean,
        std=NormalizingConstants.train_std
    )(image=image)["image"], label)
