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

        self.X, self.y = [], []
        for i in range(len(self.data)-self.window_size):
            feature = self.data[i:i+self.window_size].values.tolist()
            target = self.data[i+1:i+self.window_size+1].values.tolist()
            self.X.append(feature)
            self.y.append(target)
            

    def return_X_y(self):
        return torch.tensor(self.X), torch.tensor(self.y)


    def __len__(self):
        return len(self.data) - self.window_size


    def __getitem__(self, index):
        assert index + self.window_size < len(self.data), f'Index {index} out of range for given window size {self.window_size}'
        x, y = self.X[index], self.y[index]
        return torch.tensor(x), torch.tensor(y)
