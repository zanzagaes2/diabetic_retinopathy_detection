from utilities.transformations import *

train_transform = light_augmentation_and_normalize(
        height=512,
        width=512,
        mean=NormalizingConstants.train_mean,
        std=NormalizingConstants.train_std
)

rotate = rotate_and_normalize(
        height=512,
        width=512,
        mean=NormalizingConstants.train_mean,
        std=NormalizingConstants.train_std
)

normalize_image = normalize(
        height=512,
        width=512,
        mean=NormalizingConstants.train_mean,
        std=NormalizingConstants.train_std
)
