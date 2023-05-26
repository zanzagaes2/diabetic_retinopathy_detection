import numpy as np

import torch
from torch.utils.data import Dataset

class BlendDataset(Dataset):

    def __len__(self):
        return len(self.targets) // 2

    def __init__(self, data, predictions):
        self.predictions = predictions
        print(data)
        self.targets, self.names = data.target, data.index
        super().__init__()

    def __getitem__(self, index):
        preds = np.concatenate((self.predictions[2*index], self.predictions[2*index + 1]),
                               axis = -1, dtype = np.float32)
        targets = np.concatenate([self.targets[2*index], self.targets[2*index + 1]],
                                 axis = None, dtype = np.float32)
        names = (self.names[2*index], self.names[2*index + 1])
        return preds, targets, names
