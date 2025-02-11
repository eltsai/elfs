import torch
import torchvision
import numpy as np
import pickle

class IndexDataset(torch.utils.data.Dataset):
    """
    The dataset also return index.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        # if hasattr(self.dataset, 'targets'):
        #     return idx, (self.dataset[idx][0], self.dataset.targets[idx])
        # else:
        return idx, self.dataset[idx]

    def __len__(self):
        return len(self.dataset)