import cv2
import numpy as np
import albumentations as A

from dataset.dataset import DRDataset
from utilities.utilities import PathConstant, df_test, df_train
from tqdm import tqdm

def load_dataset(dataset = 'train', transform =  None):
    if dataset == 'train':
        return DRDataset(
            images_path=PathConstant.train_augmented_path,
            df=df_train(),
            transform=transform
        )
    if dataset == 'test':
        return DRDataset(
            images_path=PathConstant.test_augmented_path,
            df=df_test(),
            transform=transform
        )


def main(transform = None):
    mean_sum = np.array([0., 0., 0.])
    std_sum = np.array([0., 0., 0.])

    for dataset_name in ('train', 'test'):
        dataset = load_dataset(dataset = dataset_name, transform = transform)
        n = len(dataset)
        for (img, label, img_file) in tqdm(dataset):
            mean, std = cv2.meanStdDev(img)
            mean_sum += np.squeeze(mean)
            std_sum += np.squeeze(std)
        print(f'Mean of dataset {dataset_name}: {mean_sum / n}')
        print(f'Std of dataset {dataset_name}: {std_sum / n}')

if __name__ == '__main__':
    def transform(image, label):
        light_augmentation = A.Compose(
            [
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.CLAHE(p=1),
            ]
        )

        return light_augmentation(image=image)["image"], label

    main(transform = transform)
