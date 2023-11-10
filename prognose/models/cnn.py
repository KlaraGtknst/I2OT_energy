import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.model = nn.Sequential()
        pass

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred