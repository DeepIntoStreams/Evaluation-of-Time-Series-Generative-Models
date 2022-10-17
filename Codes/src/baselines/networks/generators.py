import torch
import torch.nn as nn
import numpy as np
import signatory
from src.utils import init_weights


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    # @abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        # ...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class LSTMGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int, init_fixed: bool = True):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.linear.apply(init_weights)
        # neural network to initialise h0 from the LSTM
        # we put a tanh at the end because we are initialising h0 from the LSTM, that needs to take values between [-1,1]

        self.init_fixed = init_fixed

    def forward(self, batch_size: int, n_lags: int, device: str, condition=None, z=None) -> torch.Tensor:
        if condition is not None:
            z = (0.1 * torch.randn(batch_size, n_lags,
                                   self.input_dim-condition.shape[-1])).to(device)  # cumsum(1)
            z[:, 0, :] *= 0  # first point is fixed
            z = z.cumsum(1)
            z = torch.cat([z, condition], dim=2)
        else:
            if z is None:
                z = (0.1 * torch.randn(batch_size, n_lags,
                                       self.input_dim)).to(device)  # cumsum(1)
            else:
                pass
            if self.init_fixed:
                h0 = torch.zeros(self.rnn.num_layers, batch_size,
                                 self.rnn.hidden_size).to(device)
            else:
                h0 = torch.randn(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(
                    device).requires_grad_()
            z[:, 0, :] *= 0
            z = z.cumsum(1)
        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags
        return x
