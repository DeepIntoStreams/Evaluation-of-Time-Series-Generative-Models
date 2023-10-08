import numpy as np
import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split


class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def get_var_dataset(window_size, data_size=5000, dim=3, phi=0.8, sigma=0.5):
    def multi_AR(window_size, dim=3, phi=0.8, sigma=0.5, burn_in=200):
        window_size = window_size + burn_in
        xt = np.zeros((window_size, dim))
        one = np.ones(dim)
        ide = np.identity(dim)
        MU = np.zeros(dim)
        COV = sigma * one + (1 - sigma) * ide
        W = np.random.multivariate_normal(MU, COV, window_size)
        for i in range(dim):
            xt[0, i] = 0
        for t in range(window_size - 1):
            xt[t + 1] = phi * xt[t] + W[t]
        return xt[burn_in:]

    var_samples = []
    for i in range(data_size):
        tmp = multi_AR(window_size, dim, phi=phi, sigma=sigma)
        var_samples.append(tmp)
    data_raw = torch.from_numpy(np.array(var_samples)).float()

    def get_pipeline():
        transforms = list()
        # standard scale
        transforms.append(('standard_scale', StandardScalerTS(axis=(0, 1))))
        pipeline = Pipeline(steps=transforms)
        return pipeline

    pipeline = get_pipeline()
    data_preprocessed = pipeline.transform(data_raw)
    return data_preprocessed


class VAR(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        **kwargs,
    ):
        n_lags = n_lags

        data_loc = pathlib.Path(
            'data/VAR/processed_data_{}'.format(n_lags))

        if os.path.exists(data_loc):
            # if data file exists pass
            pass
        else:
            # else generate path
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = get_var_dataset(
                window_size=n_lags, data_size=20000, dim=3, phi=0.8, sigma=0.5)
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(VAR, self).__init__(X)

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
