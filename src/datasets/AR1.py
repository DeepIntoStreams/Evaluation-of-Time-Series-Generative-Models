
import numpy as np

import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split


class AROne:
    '''
    :param D: dimension of x
    :param T: sequence length
    :param phi: parameters for AR model
    :param s: parameter that controls the magnitude of covariance matrix
    '''

    def __init__(self, D=3, T=30, phi=np.linspace(0.1, 0.3, 3), s=0.5, burn=10):
        self.D = D
        self.T = T
        self.phi = phi
        self.Sig = np.eye(D) * (1 - s) + s
        self.chol = np.linalg.cholesky(self.Sig)
        self.burn = burn

    def batch(self, N):
        x0 = np.random.randn(N, self.D)
        x = np.zeros((self.T + self.burn, N, self.D))
        x[0, :, :] = x0
        for i in range(1, self.T + self.burn):
            x[i, ...] = self.phi * x[i - 1] + \
                np.random.randn(N, self.D) @ self.chol.T

        x = x[-self.T:, :, :]
        x = np.swapaxes(x, 0, 1)
        return x.astype("float32")


class AR1_dataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        **kwargs,
    ):
        n_lags = n_lags

        data_loc = pathlib.Path(
            'data/AR1/processed_data_{}'.format(n_lags))

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = AROne(T=n_lags).batch(10000)
            x_real = torch.from_numpy(x_real).float()
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(AR1_dataset, self).__init__(X)

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
