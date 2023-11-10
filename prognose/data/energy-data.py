""" Custom PyTorch Dataset for energy data """

import torch
from torch.utils.data import Dataset
from torchvision.transforms import * 


class EnergyDataset(Dataset):

    def __init__(self):
        super(EnergyDataset, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass