import pandas as pd
import numpy as np
import albumentations as A
import os

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from PIL import Image

@dataclass
class DRDataset(Dataset):
    
    images_path : str
    df : pd.DataFrame
    transform : A.BasicTransform = None
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_file, label  = self.df.iloc[index]
        image = np.array(Image.open(os.path.join(self.images_path, image_file)), dtype = np.float64)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label, image_file

@dataclass
class DRDatasetPairs(Dataset):
    images_path: str
    df: pd.DataFrame
    transform: A.BasicTransform = None

    def __len__(self):
        return len(self.df) // 2

    def __getitem__(self, index):
        index *= 2
        file1, target1 = self.df.iloc[index]
        file2, target2 = self.df.iloc[index+1]

        image1 = np.array(Image.open(os.path.join(self.images_path, file1)), dtype=np.float64)
        image2 = np.array(Image.open(os.path.join(self.images_path, file2)), dtype=np.float64)

        if self.transform:
            image1, target1 = self.transform(image1, target1)
            image2, target2 = self.transform(image2, target2)
        return np.stack((image1, image2)).astype(np.float32),\
            np.array([target1, target2]), \
            (file1, file2)
