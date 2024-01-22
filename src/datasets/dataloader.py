import pandas
import torch
import ml_collections
from typing import Tuple
from torch.utils.data import DataLoader
from src.datasets.rough import Rough_S
from src.datasets.sMnist import MNIST
from src.datasets.stock import Stock
from src.datasets.beijing_air_quality import Beijing_air_quality
from src.datasets.AR1 import AR1_dataset
from src.datasets.GBM import GBM
from src.datasets.Custom import Custom_Dataset


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    shuffle: bool = True,
    custom_dataset = None
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """


    if custom_dataset == None:
        dataset = {
            "ROUGH": Rough_S,
            # "MNIST": MNIST,
            "STOCK": Stock,
            "Air_Quality": Beijing_air_quality,
            "AR1": AR1_dataset,
            "GBM": GBM


        }[config.dataset]

        data_dir = config.data_dir + config.dataset + '/processed_data_{}'.format(config.n_lags)

        training_set = dataset(
            partition="train",
            n_lags=config.n_lags,
            data_dir=data_dir
        )
        test_set = dataset(
            partition="test",
            n_lags=config.n_lags,
            data_dir=data_dir
        )
    else:
        dataset = Custom_Dataset
        config.update({"n_lags": custom_dataset.shape[1]}, allow_val_change=True)

        training_set = dataset(
            partition="train",
            n_lags=config.n_lags,
            dataset=custom_dataset,
            dataset_name=config.dataset,
        )
        test_set = dataset(
            partition="test",
            n_lags=config.n_lags,
            dataset=custom_dataset,
            dataset_name=config.dataset,
        )


    training_loader = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    config.update({"n_lags": next(iter(test_loader))[
                  0].shape[1]}, allow_val_change=True)
    print("data shape:", next(iter(test_loader))[0].shape)

    if config.conditional:
        config.update({"input_dim": training_loader.dataset[0][0].shape[-1] + config.num_classes}, allow_val_change=True)
        # config.input_dim = training_loader.dataset[0][0].shape[-1] + \
        #     config.num_classes
    else:
        config.update({"input_dim": training_loader.dataset[0][0].shape[-1]}, allow_val_change = True)
        # config.input_dim = training_loader.dataset[0][0].shape[-1]
    return training_loader, test_loader


if __name__ == '__main__':
    from src.datasets.dataloader import get_dataset
    import yaml
    import ml_collections
    config_dir = 'configs/' + 'train_gan.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    train_dl, test_dl = get_dataset(config=config)
