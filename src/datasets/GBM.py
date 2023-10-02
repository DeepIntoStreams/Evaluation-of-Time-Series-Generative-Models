from tqdm import tqdm
import numpy as np
from fbm import fbm, MBM
import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split


def get_GBM_paths(drift=0., scale=0.1, dim=2, size=20000, n_lags=50, h=1):
    r"""
    Paths of Geometric Brownian Motion 

    dS_t = \mu S_t dt + \sigma S_t dW_t

    where W_t denotes the Brownian motion
    
    Parameters
    ----------
    drift: float
    scale: float
    dim: int
    size: int
        size of the dataset
    n_lags: int
        Number of timesteps in the path

    Returns
    -------
    dataset: torch.tensor
        array of shape (size, n_lags, dim)

    """

    print('Generate 20000 path samples of geometric Brownian motion')
    dataset = torch.ones(size, n_lags, dim)
    dataset[:, 1:, :] = torch.exp(
        (drift-scale**2/2)*h + (scale*np.sqrt(h)*torch.randn(size, n_lags-1, dim)))
    dataset = dataset.cumprod(1)
    return dataset



class GBM(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        **kwargs,
    ):
        n_lags = n_lags

        data_loc = pathlib.Path(
            'data/GBM/processed_data_{}'.format(n_lags))

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = get_GBM_paths(n_lags=n_lags)
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(GBM, self).__init__(X)

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

        return X
