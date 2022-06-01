from argparse import ArgumentParser, _ArgumentGroup
from typing import Union, Any, TypeVar
import torch
import numpy as np
import os


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(
        dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()


def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float
):
    """
    Apply a train-test split to a given tensor along the first dimension of the tensor.

    Parameters
    ----------
    x: torch.Tensor, tensor to split.
    train_test_ratio, percentage of samples kept in train set, i.e. 0.8 => keep 80% of samples in the train set

    Returns
    -------
    x_train: torch.Tensor, training set
    x_test: torch.Tensor, test set
    """
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size)
    indices_test = torch.LongTensor(
        [i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


class WithDefaultsWrapper:
    """
    Automatically add default values to argument help text.
    Wrapper for argument parser (and ArgumentGroups) to automatically add
    default values at the end of the help text (and automatically wrap any
    subgroup with this wrapper).
    """

    def __init__(
        self, argument_parser: Union["WithDefaultsWrapper", ArgumentParser,
                                     _ArgumentGroup]
    ) -> None:
        """
        Parameters
        ----------
        argument_parser : ArgumentParser or _ArgumentGroup object
        """
        self._argument_parser = argument_parser

    def __getattr__(self, name: str) -> Any:
        """
        Pass access of everything not definedhere directly on to the parser.
        """
        return getattr(self._argument_parser, name)

    def add_argument(self, *args, **kwargs) -> None:
        """
        Add default value to end of help text.
        If help text and default value are defined, appends the default value
        to the help text and passes everything on to the argument parser object
        """
        if ("help" in kwargs and "default" in kwargs
                and kwargs["default"] != "==SUPPRESS=="
                and kwargs["default"] is not None):
            kwargs["help"] += " (default is {})".format(kwargs["default"])
        self._argument_parser.add_argument(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs) -> "WithDefaultsWrapper":
        """
        Wrap new subgroups with this wrapper.
        """
        group = self._argument_parser.add_argument_group(*args, **kwargs)
        group = WithDefaultsWrapper(group)
        return group


ParserType = TypeVar("ParserType", ArgumentParser, WithDefaultsWrapper)
