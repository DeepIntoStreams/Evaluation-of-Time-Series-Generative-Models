
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
