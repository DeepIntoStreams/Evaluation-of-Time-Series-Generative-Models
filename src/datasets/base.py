import torch
import os
from src.datasets.utils import load_data, save_data, train_test_split

class BaseDataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        data_dir: str,
        **kwargs,
    ):
        self.n_lags = n_lags
        self.data_loc = data_dir
        self.partition = partition

        if os.path.exists(self.data_loc):
            pass
        else:
            if not os.path.exists(self.data_loc.parent):
                os.mkdir(self.data_loc.parent)
            if not os.path.exists(self.data_loc):
                os.mkdir(self.data_loc)
            x_real = self.get_paths()
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                self.data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(self.data_loc, partition)
        super(BaseDataset, self).__init__(X)

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

    def get_paths(self):
        raise NotImplementedError("Data generation not implemented")