from abc import ABC, abstractmethod
import numpy as np
import joblib
import torch
from torch import nn, einsum
from torch.optim import Adam
from collections import OrderedDict


class BaseVariationalAutoencoder(nn.Module, ABC):
    def __init__(self,
                 n_lags,
                 input_dim,
                 latent_dim,
                 reconstruction_wt=3.0,
                 **kwargs):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt

        self.encoder = None
        self.decoder = None

    def forward(self, X):
        raise NotImplementedError
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def get_prior_samples(self, num_samples):
        Z = torch.randn([num_samples, self.latent_dim])
        samples = self.decoder.predict(Z)
        return samples

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError


class VariationalAutoencoderConvInterpretable(BaseVariationalAutoencoder):

    def __init__(self, hidden_layer_sizes, trend_poly=0, num_gen_seas=0, custom_seas=None,
                 use_scaler=False, use_residual_conn=True, **kwargs):
        '''
            hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder.
            trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
            num_gen_seas: Number of sine-waves to use to model seasonalities. Each sine wae will have its own amplitude, frequency and phase.
            custom_seas: list of tuples of (num_seasons, len_per_season).
            num_seasons: number of seasons per cycle.
            len_per_season: number of epochs (time-steps) per season.
            use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
            trend, generic and custom seasonalities.
        '''

        super(VariationalAutoencoderConvInterpretable, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.num_gen_seas = num_gen_seas
        self.custom_seas = custom_seas
        self.use_scaler = use_scaler
        self.use_residual_conn = use_residual_conn

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_encoder(self):
        encoder = encoder_block(input_dim=self.input_dim,
                                latent_dim=self.latent_dim,
                                n_lags=self.n_lags,
                                hidden_layer_sizes=self.hidden_layer_sizes)

        return encoder

    def _get_decoder(self):
        decoder = decoder_block(input_dim=self.input_dim,
                                latent_dim=self.latent_dim,
                                n_lags=self.n_lags,
                                hidden_layer_sizes=self.hidden_layer_sizes,
                                trend_poly=self.trend_poly,
                                num_gen_seas=self.num_gen_seas,
                                use_scaler=self.use_scaler,
                                custom_seas=self.custom_seas,  # tuple consisting of pairs (num_seasons, len_per_season)
                                use_residual_conn=self.use_residual_conn
                                )
        return decoder

    def forward(self, batch_size: int, n_lags: int, device: str, condition=None, z=None):
        if condition is not None:
            Z = torch.randn([batch_size, self.latent_dim]).to(device)
            Z = torch.cat([Z, condition], dim=-1)
        else:
            if z is None:
                Z = torch.randn([batch_size, self.latent_dim]).to(device)
            else:
                Z = z.to(device)
        x = self.decoder(Z)
        return x


class encoder_block(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_lags: int, hidden_layer_sizes: list):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential()
        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            self.encoder.add_module("conv_{}".format(i),
                                    torch.nn.Conv1d(in_channels=self.input_dim,
                                                    out_channels=hidden_size,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1
                                                    ))
            self.input_dim = hidden_size
            self.encoder.add_module('norm_{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.encoder.add_module('ReLU_{}'.format(i),
                                    torch.nn.ReLU())

        # Get the last dimension after the convolution
        final_length = conv_length(self.n_lags, len(hidden_layer_sizes)) * self.hidden_layer_sizes[-1]

        self.encoder_mu = nn.Linear(final_length, self.latent_dim)
        self.encoder_log_var = nn.Linear(final_length, self.latent_dim)

        self.sampler = sampling_layer()

    def forward(self, z):
        '''
        Input dimension: [Batch, input_dim, n_lags]
        Output dimension: [Batch, latent_dim]
        '''
        batch = z.shape[0]
        hidden_state = self.encoder(z).reshape(batch, -1)

        z_mean = self.encoder_mu(hidden_state)
        z_log_var = self.encoder_log_var(hidden_state)

        encoder_output = self.sampler(z_mean, z_log_var)

        return encoder_output, z_mean, z_log_var


class sampling_layer(nn.Module):
    def __init__(self):
        super(sampling_layer, self).__init__()

    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def forward(self, z_mean, z_log_var):
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn([batch, dim]).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


def conv_length(init_length, number_of_layers, kernel=3, stride=2, padding=1):
    for i in range(number_of_layers):
        init_length = int((init_length + 2 * padding - kernel) / 2 + 1)
    return init_length


def inverse_conv_length(init_length, number_of_layers, kernel=3, stride=2, padding=1):
    print(init_length)
    for i in range(number_of_layers):
        init_length = int(((init_length - 1) * stride - 2 * padding + kernel))
        print(init_length)
    return init_length


# Decoder

class decoder_block(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 n_lags,
                 hidden_layer_sizes,
                 trend_poly=0,
                 num_gen_seas=0,
                 use_scaler=False,
                 custom_seas=None,  # tuple consisting of pairs (num_seasons, len_per_season)
                 use_residual_conn=True
                 ):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim  # input_dimension
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.num_gen_seas = num_gen_seas
        self.custom_seas = custom_seas
        self.use_scaler = use_scaler
        self.use_residual_conn = use_residual_conn

        self.level_model = level_block(self.input_dim, self.latent_dim, self.n_lags)

        # trend polynomials
        if self.trend_poly is not None and self.trend_poly > 0:
            self.trend_model = trend_block(self.input_dim, self.latent_dim, self.n_lags, self.trend_poly)

        # # generic seasonalities
        if self.num_gen_seas is not None and self.num_gen_seas > 0:
            self.generic_model = generic_seasonal_block(self.input_dim,
                                                        self.latent_dim,
                                                        self.n_lags,
                                                        self.num_gen_seas)
        #  #  self.generic_model = generic_seasonal_block2(self.input_dim,
    #                                                             self.latent_dim,
    #                                                             self.n_lags,
    #                                                             self.num_gen_seas)

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            self.custom_model = custom_seasonal_block(self.input_dim,
                                                      self.latent_dim,
                                                      self.n_lags,
                                                      self.custom_seas)

        # residual_block
        if self.use_residual_conn:
            self.residual_model = decoder_residual_block(self.input_dim,
                                                         self.latent_dim,
                                                         self.n_lags,
                                                         self.hidden_layer_sizes)
        # scaling block
        if self.use_scaler:
            self.scale_model = level_block(self.input_dim, self.latent_dim, self.n_lags)

    def forward(self, z):
        '''
        Input dimension: [Batch, latent_dim]
        Output dimension: [Batch, input_dim, n_lags]
        '''
        batch = z.shape[0]

        output = self.level_model(z)

        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = self.trend_model(z)
            output += trend_vals

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            custrom_seas_vals = self.custom_model(z)
            output += custrom_seas_vals

        if self.use_residual_conn:
            residuals = self.residual_model(z)
            output += residuals

        if self.use_scaler and output is not None:
            scale = self.scale_model(z)
            output *= scale

        if output is None:
            raise Exception('''Error: No decoder model to use. 
            You must use one or more of:
            trend, generic seasonality(ies), custom seasonality(ies), and/or residual connection. ''')

        return output.permute([0, 2, 1])


class level_block(nn.Module):
    def __init__(self, input_dim, latent_dim, n_lags):
        super().__init__()
        self.n_lags = n_lags
        self.model = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, z):
        """
        Input has shape [batch, latent_dim]
        """
        output = self.model(z).unsqueeze(1)  # (Batch, 1, input_dim)
        output = output.repeat((1, self.n_lags, 1))  # (Batch, T, input_dim)
        #         ones = torch.ones([1, self.n_lags, 1])
        #         output = output * ones
        return output


class trend_block(nn.Module):
    def __init__(self, input_dim, latent_dim, n_lags, trend_poly=2):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.trend_poly = trend_poly
        self.model = nn.Sequential(
            nn.Linear(latent_dim, input_dim * trend_poly),
            nn.LeakyReLU(),
            nn.Linear(input_dim * trend_poly, input_dim * trend_poly),
        )

    def forward(self, z):
        """
        Input has shape [Batch, latent_dim]
        """
        trend_params = self.model(z)  # (Batch, input_dim)
        trend_params = torch.reshape(trend_params, (trend_params.shape[0], self.input_dim, self.trend_poly))  # (Batch, input_dim, P)

        lin_space = (torch.arange(0, self.n_lags, 1) / self.n_lags).to(z.device)

        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0)  # shape: (P, T)

        trend_vals = torch.einsum('bki, ij -> bkj', trend_params, poly_space).permute([0, 2, 1])  # shape (Batch, T, input_dim)

        return trend_vals


class custom_seasonal_block(nn.Module):
    def __init__(self, input_dim, latent_dim, n_lags, custom_seas):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.custom_seas = custom_seas  # list of tuples of (num_seasons, len_per_season).
        self.model = nn.Sequential()

        for i, season_tup in enumerate(self.custom_seas):
            num_seasons, len_per_season = season_tup
            self.model.add_module('linear_{}'.format(i), nn.Linear(latent_dim, input_dim * num_seasons))

    def forward(self, z):
        """
        Input has shape [Batch, latent_dim]
        """
        batch_size = z.shape[0]
        ones = torch.ones([batch_size, self.input_dim, self.n_lags])

        all_seas_vals = []
        for i, season_tup in enumerate(self.custom_seas):
            num_seasons, len_per_season = season_tup
            temp_season = self.model[i](z).reshape(batch_size, self.input_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(num_seasons, len_per_season)  # (T, )
            #             dim2_idxes = ones * torch.reshape(season_indexes_over_time, shape=(1,1,-1))

            season_indexes_over_time = torch.reshape(torch.Tensor(season_indexes_over_time), (1, 1, -1)).long()  # (1, 1, T)

            dim2_idxes = season_indexes_over_time.repeat([batch_size, self.input_dim, 1])  # (Batch, input_dim, T)

            season_vals = torch.gather(temp_season, -1, dim2_idxes)  # (Batch, input_dim, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, axis=-1)  # (Batch, input_dim, T, S)
        all_seas_vals = torch.sum(all_seas_vals, axis=-1).permute([0, 2, 1])  # (Batch, T, input_dim)

        return all_seas_vals

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        curr_len = 0
        season_idx = []
        curr_idx = 0
        while curr_len < self.n_lags:
            reps = len_per_season if curr_len + len_per_season <= self.n_lags else self.n_lags - curr_len
            season_idx.extend([curr_idx] * reps)
            curr_idx += 1
            if curr_idx == num_seasons: curr_idx = 0
            curr_len += reps
        return season_idx


class generic_seasonal_block(nn.Module):
    def __init__(self, input_dim, latent_dim, n_lags, num_gen_seas=1):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_gen_seas = num_gen_seas
        self.model = nn.Sequential(OrderedDict([
            ('g_season_freq', nn.Linear(self.latent_dim, self.input_dim * self.num_gen_seas)),
            ('g_season_phase', nn.Linear(self.latent_dim, self.input_dim * self.num_gen_seas)),
            ('g_season_amplitude', nn.Linear(self.latent_dim, self.input_dim * self.num_gen_seas))
        ]))

    def forward(self, z):
        """
        Input has shape [Batch, latent_dim]
        """
        freq = self.model[0](z).reshape(z.shape[0], 1, self.input_dim, self.num_gen_seas)  # shape: (Batch, 1, input_dim, S)
        phase = self.model[1](z).reshape(z.shape[0], 1, self.input_dim, self.num_gen_seas)  # shape: (Batch, 1, input_dim, S)
        amplitude = self.model[2](z).reshape(z.shape[0], 1, self.input_dim, self.num_gen_seas)  # shape: (Batch, 1, input_dim, S)

        lin_space = (torch.arange(0, self.n_lags, 1) / self.n_lags).reshape(1, -1, 1, 1).to(z.device)  # shape: (1, T, 1, 1)
        seas_vals = (amplitude * torch.sin(2. * np.pi * freq * lin_space + phase)).sum(-1)  # shape: (Batch, T, input_dim)

        return seas_vals, freq, phase, amplitude


class generic_seasonal_block2(nn.Module):
    def __init__(self, input_dim, latent_dim, n_lags, num_gen_seas):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_gen_seas = num_gen_seas
        self.model = nn.Linear(self.latent_dim, self.input_dim * self.num_gen_seas)

    def forward(self, z):
        """
        Input has shape [Batch, latent_dim]
        """
        season_params = self.model(z).reshape(z.shape[0], self.input_dim, self.num_gen_seas)  # shape: (Batch, input_dim, S)

        p = self.num_gen_seas
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)

        lin_space = (torch.arange(0, self.n_lags, 1) / self.n_lags)  # shape: (T, )

        s1 = torch.stack([torch.cos(2 * np.pi * i * lin_space) for i in range(p1)], axis=0)
        s2 = torch.stack([torch.sin(2 * np.pi * i * lin_space) for i in range(p2)], axis=0)

        if p == 1:
            s = s2
        else:
            s = torch.cat([s1, s2], axis=0)

        seas_vals = torch.einsum('bij, jt -> bit', season_params, s).permute([0, 2, 1])  # shape: (N, T, D)

        return seas_vals


class decoder_residual_block(nn.Module):
    def __init__(self, input_dim, latent_dim, n_lags, hidden_layer_sizes):
        super().__init__()
        self.n_lags = n_lags
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.init_channel = self.hidden_layer_sizes[-1]
        self.init_length = conv_length(self.n_lags, len(hidden_layer_sizes)) * self.init_channel
        self.init_length_ = conv_length(self.n_lags, len(hidden_layer_sizes))
        self.final_length = inverse_conv_length(self.init_length_, len(hidden_layer_sizes))

        self.first_linear_layer = nn.Linear(latent_dim, self.init_length)

        self.model = torch.nn.Sequential()
        for i, hidden_size in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            self.model.add_module("conv_{}".format(i),
                                  torch.nn.ConvTranspose1d(in_channels=self.init_channel,
                                                           out_channels=hidden_size,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1
                                                           ))
            self.init_channel = hidden_size
            self.model.add_module('norm_{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.model.add_module('ReLU_{}'.format(i),
                                  torch.nn.ReLU())

        self.model.add_module("last_conv",
                              torch.nn.ConvTranspose1d(in_channels=hidden_size,
                                                       out_channels=self.input_dim,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1
                                                       ))
        self.model.add_module('last_ReLU_{}',
                              torch.nn.ReLU())

        self.last_linear_layer = nn.Linear(self.final_length * self.input_dim, self.n_lags * self.input_dim)

    def forward(self, z):
        """
        Input dimension: [Batch, latent_dim]
        """
        batch = z.shape[0]
        z = self.first_linear_layer(z).reshape([batch, self.hidden_layer_sizes[-1], -1])  # (Batch, t, hidden_dim)
        z = self.model(z).flatten(1)  # (Batch, )
        z = self.last_linear_layer(z).reshape([batch, -1, self.input_dim])  # (Batch, T, input_dim)
        return z