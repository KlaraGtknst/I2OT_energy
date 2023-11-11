""" Custom PyTorch Dataset for energy data """

import torch
from torch.utils.data import Dataset
from torchvision.transforms import * 
import pandas as pd


class EnergyDataset(Dataset):

    def __init__(self, path, window_size):
        super(EnergyDataset, self).__init__()
        self.data = pd.read_csv(path)
        self.data = self.data[['_value']]
        self.window_size = window_size
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index < (len(self.data) + self.window_size), f'Index {index} out of range for given window size {self.window_size}'
        x = self.data[index:index+self.window_size].values.tolist()
        y = self.data[index+1:index+self.window_size+1].values.tolist()
        return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)