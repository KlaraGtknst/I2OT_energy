import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred


    