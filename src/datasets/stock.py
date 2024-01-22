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


class Stock(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):
        n_lags = kwargs["n_lags"]
        self.root = pathlib.Path('data')

        if 'n_lags' in kwargs:
            data_loc = pathlib.Path(
                'data/stock/processed_data_{}'.format(kwargs['n_lags']))
        else:
            data_loc = pathlib.Path(
                'data/stock/processed_data')

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = self._process_data(n_lags=n_lags)
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(Stock, self).__init__(X)

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
        base_loc = root / "stock"
        loc = base_loc / "oxfordmanrealizedvolatilityindices.zip"
        if os.path.exists(loc):
            return
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            'https://github.com/onnokleen/mfGARCH/raw/v0.1.9/data-raw/OxfordManRealizedVolatilityIndices.zip',
            str(loc),
        )

        with zipfile.ZipFile(loc, "r") as f:
            f.extractall(str(base_loc))

    def _process_data(self, n_lags):
        data_loc = pathlib.Path(
            'data/stock/oxfordmanrealizedvolatilityindices.csv')
        oxford = pd.read_csv(data_loc
                             )
        start = '2005-01-01 00:00:00+01:00'
        end = '2020-06-01 00:00:00+01:00'
        vol_type = 'medrv'
        assets = [".SPX"]
        # collect ticker dfs
        dfs = list()
        for ticker in assets:
            dfs.append(
                oxford[oxford['Symbol'] == ticker].set_index(['Unnamed: 0'])[
                    start:end]
            )

        # find index intersections between ticker dfs
        index = dfs[0].index
        for df, symbol in zip(dfs, assets):
            index = index.intersection(df.index)

        vols = list()
        logrtns = list()
        for df in dfs:
            spot = df.loc[index][['close_price']].values
            logrtn = np.log(spot[1:]) - np.log(spot[:-1])
            logrtns.append(logrtn)
            vols.append(df.loc[index][[vol_type]].values)

        logrtns = np.concatenate(logrtns, axis=1)

        vols = np.concatenate(vols, axis=1)
        vols[vols <= 1e-6] = 1e-6
        vols = np.log(vols)

        x = np.concatenate([logrtns, vols[1:]], axis=1)
        x = torch.from_numpy(x).float().unsqueeze(0)
        pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
        data_preprocessed = pipeline.transform(x)
        # we only learn the logreturn
        return rolling_window(data_preprocessed[..., :], n_lags)