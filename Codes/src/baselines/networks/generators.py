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
                if self.init_fixed:
                    h0 = torch.zeros(self.rnn.num_layers, batch_size,
                                     self.rnn.hidden_size).to(device)
            else:
                pass
                h0 = torch.randn(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(
                    device).requires_grad_()
            z[:, 0, :] *= 0
            z = z.cumsum(1)
        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags
        return x

    
class CVAE(GeneratorBase):
    """Conditional Variational Auto Encoder (CVAE)."""
    
    def __init__(self, input_dim: int, output_dim: int, dim: int, level: int, latent_dim: int, augmentation: str = None, hidden_dim: int = 50, alpha:: float = 0.2):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dim = dim # path dimension
        self.level = level # level up to which the logsignature is calculated
        self.alpha = alpha
        self.augmentation = augmentation
        if self.augmentation == 'lead_lag':
            self.aug_dim = 2*self.dim
        else:
            self.aug_dim = self.dim
        self.input_dim = signatory.logsignature_channels(self.aug_dim, level) # feat_dimension: logsignature dimension
        
        self.encoder = encoder_block(self.feat_dim, self.latent_dim, self.hidden_dim)
        self.decoder = decoder_block(self.feat_dim, self.latent_dim, self.hidden_dim)
        
        def forward(self, cond, device):
            batch_size = cond.shape[0]
            noise = torch.randn([batch_size, latent_dim])
            x_fake = self.decoder(noise, cond)
            return x_fake
        
    
class encoder_block(nn.Module):
    """Encoder block"""
    def __init__(self, feat_dim, latent_dim, hidden_dim = 50):
        super().__init__()
        self.feat_dim = feat_dim # feat_dimension: logsignature dimension
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        
        self.model_x = nn.Sequential(
            nn.Linear(self.feat_dim + 1, self.hidden_dim),
            nn.LeakyReLU(0.3)
        )
        
        self.model_mean = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.LeakyReLU(0.3)
        )
        
        self.model_std = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.LeakyReLU(0.3)
        )
    
    def forward(self, X_in, cond):
        input_x = torch.cat([X_in, cond], dim = -1)
        input_x = self.model_x(input_x)
        
        mean = self.model_mean(input_x)
        log_var = self.model_std(input_x)
        
        epsilon = torch.randn([input_x.shape[0], self.latent_dim]).to(mean.device)
#         print(mean, epsilon, log_var)
        z = mean + epsilon * torch.exp(log_var / 2.)
        
        return z, mean, log_var
    
class decoder_block(nn.Module):
    """Decoder block"""
    def __init__(self, feat_dim, latent_dim, hidden_dim = 50):
        super().__init__()
        self.feat_dim = feat_dim # feat_dimension
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        
        self.model_z = nn.Sequential(
            nn.Linear(self.latent_dim + 1, self.hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(self.hidden_dim, self.feat_dim),
            nn.Sigmoid(),
        )
        
    
    def forward(self, Z_in, cond):
        input_z = torch.cat([Z_in, cond], dim = -1)
        z = self.model_z(input_z)
        return z