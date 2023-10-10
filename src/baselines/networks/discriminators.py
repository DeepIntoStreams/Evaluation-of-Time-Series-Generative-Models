
import torch
from torch import nn
from typing import Tuple


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, out_dim=1, return_seq=False):
        super(LSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.return_seq = return_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_seq:
            h = self.lstm(x)[0]
        else:
            h = self.lstm(x)[0][:, -1:]
        x = self.linear(h)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size=2):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers,
                            hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.rnn(x)[0][:, -1]
        return self.linear(x)