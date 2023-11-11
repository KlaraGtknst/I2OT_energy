import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_pred, _ = self.lstm(x)
        y_pred = self.linear(y_pred)
        return y_pred


    