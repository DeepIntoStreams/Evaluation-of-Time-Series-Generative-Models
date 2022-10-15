import numpy as np
import torch
import pathlib
import os
import urllib.request
import zipfile
from .utils import load_data, save_data, train_test_split
import pandas as pd
import glob


class Beijing_air_quality(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):
        self.root = pathlib.Path('data')

        data_loc = pathlib.Path(
            'data/air_quality/processed_data')

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = self._process_data()
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(Beijing_air_quality, self).__init__(X)

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

    def download(self):
        root = self.root
        base_loc = root / "air_quality"
        loc = base_loc / "PRSA2017_Data_20130301-20170228.zip"
        if os.path.exists(loc):
            return
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip',
            str(loc),
        )

        with zipfile.ZipFile(loc, "r") as f:
            f.extractall(str(base_loc))

    @staticmethod
    def remove_nan_sample(X: torch.tensor):
        tensor = X.reshape(X.shape[0], -1)
        tensor = tensor[~torch.any(tensor.isnan(), dim=1)]
        return tensor.reshape(-1, X.shape[1], X.shape[2])

    def _process_data(self):
        data_loc = glob.glob(
            'data/air_quality/PRSA_Data_20130301-20170228/*csv')
        df = pd.concat([pd.read_csv(f) for f in data_loc], 0)
        columns_interest = ['SO2', 'NO2', 'CO', 'O3', 'PM2.5', 'PM10']
        dataset = []
        for idx, (ind, group) in enumerate(df.groupby(['year', 'month', 'day', 'station'])):
            dataset.append(group[columns_interest].values)
        dataset = np.stack(dataset, axis=0)
        dataset = torch.from_numpy(dataset).float()

        return self.remove_nan_sample(dataset)
