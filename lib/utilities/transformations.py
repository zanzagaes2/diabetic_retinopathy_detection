import albumentations as A
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2

class NormalizingConstants:
    # Calculated with the stats module over the training dataset
    train_mean = np.array([128.0, 128.0, 128.0])
    train_std = np.array([19.306323, 21.389749, 15.551280])

def normalize(height, width, mean, std):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=1.0,
            p = 1
        ),
        ToTensorV2()
    ])

def rotate_and_normalize(width, height, mean, std):
    return A.Compose([
        A.Affine(rotate=(0, 360)),
        A.Resize(width=width, height=height, interpolation=cv2.INTER_AREA),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=1.0,
            p = 1
        ),
        ToTensorV2(),
    ])

def light_augmentation_and_normalize(width, height, mean, std):
    return A.Compose([
        A.Affine(scale=(.9, 1.1)),
        A.Affine(rotate=(0, 360)),
        A.Flip(p = 0.5),
        A.Resize(width=width, height=height, interpolation=cv2.INTER_AREA),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=1.0,
            p = 1
        ),
        ToTensorV2(),
    ])
