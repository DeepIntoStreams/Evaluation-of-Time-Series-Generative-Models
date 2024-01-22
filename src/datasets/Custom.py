import numpy as np
import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split
from src.datasets.Pipeline import Pipeline, StandardScalerTS
import pandas as pd
import urllib.request
import zipfile


def rolling_window(x: torch.Tensor, n_lags):
    return torch.cat([x[:, t:t + n_lags] for t in range(x.shape[1] - n_lags + 1)], dim=0)


class Custom_Dataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        dataset,
        dataset_name,
        **kwargs,
    ):
        n_lags = kwargs["n_lags"]
        self.root = pathlib.Path('data')

        if 'n_lags' in kwargs:
            data_loc = pathlib.Path(
                'data/{}/processed_data_{}'.format(dataset_name, kwargs['n_lags']))
        else:
            data_loc = pathlib.Path(
                'data/{}/processed_data'.format(dataset_name))

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = self._process_data(dataset, n_lags=n_lags)
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(Custom_Dataset, self).__init__(X)

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

    def _process_data(self, dataset, n_lags):
        x = dataset.float().unsqueeze(0)
        pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
        data_preprocessed = pipeline.transform(x)
        return data_preprocessed