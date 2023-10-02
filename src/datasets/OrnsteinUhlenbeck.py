import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split


def get_OU_data(dataset_size=8192, t_size=64, withtime=False):
    '''
    Input:
    dataset_size: int, the size of the generated dataset
    t_size: int, the number of time points generated 
    withtime: bool, gnerate OU path with time or not.
    Output:
    ys: the OU path with shape (dataset_szie,t_size, 2) if withtime.
    '''
    class OrnsteinUhlenbeckSDE(torch.nn.Module):
        sde_type = 'ito'
        noise_type = 'scalar'

        def __init__(self, mu, theta, sigma):
            super().__init__()
            self.register_buffer('mu', torch.as_tensor(mu))
            self.register_buffer('theta', torch.as_tensor(theta))
            self.register_buffer('sigma', torch.as_tensor(sigma))

        def f(self, t, y):
            return self.mu * t - self.theta * y

        def g(self, t, y):
            return self.sigma.expand(y.size(0), 1, 1) * (2 * t / t_size)

    ou_sde = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4).to(device)
    y0 = torch.rand(dataset_size, device=device).unsqueeze(-1) * 2 - 1
    ts = torch.linspace(0, t_size-1, t_size, device=device)
    ys = torchsde.sdeint(ou_sde, y0, ts, dt=1e-1)

    ###################
    # Typically important to normalise data. Note that the data is normalised with respect to the statistics of the
    # initial data, _not_ the whole time series. This seems to help the learning process, presumably because if the
    # initial condition is wrong then it's pretty hard to learn the rest of the SDE correctly.
    ###################
    y0_flat = ys[0].view(-1)
    y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat))
    ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()

    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    if withtime:
        ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1),
                        ys.transpose(0, 1)], dim=2)

    return ys


class OU(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        **kwargs,
    ):
        n_lags = n_lags

        data_loc = pathlib.Path(
            'data/OU/processed_data_{}'.format(n_lags))

        if os.path.exists(data_loc):
            # if data file exists pass
            pass
        else:
            # else generate path
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            # get_var_dataset(window_size=n_lags, batch_size=5000, dim=3, phi=0.8, sigma=0.5)
            x_real = get_OU_data(dataset_size=20000,
                                 t_size=n_lags, withtime=False)
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(OU, self).__init__(X)

    @staticmethod
    def load_data(data_loc, partition):
        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
        elif partition == "test":
            X = tensors["test_X"]
        else:
            raise NotImplementedError(
                "the set {} is not implemented.".format(set))
        return
