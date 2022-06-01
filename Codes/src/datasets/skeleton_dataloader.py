from typing import Optional

from src.datasets.utils import WithDefaultsWrapper, ParserType
from torch.utils.data import Dataset
import cv2
import numpy as np
from warnings import warn
from typing import Sequence, Optional, Tuple, Union
from abc import ABC


class DataSubset:
    """
    Provides a Sequence for a given subset of a DatasetLoader object.
    Sequence object provide __len__ and __getitem__ and so can be directly
    passed into a PyTorch DatasetLoader.
    """

    def __init__(self, dataset_loader, split_name, subset):

        self._dataset_loader = dataset_loader
        self._samples = self._dataset_loader.get_split(split_name, subset)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        sample = self._dataset_loader[self._samples[index]]
        return tuple(sample[col]
                     for col in self._dataset_loader._selected_cols)


class DatasetLoader(ABC):
    """
    Base class for all dataset loaders to provide a common interface for
    retrieving the data out of the dataset object.
    """
    _general_parser_args_added = False
    _parser_split_added = False

    def __init__(self, no_lazy_loading=False, split=None, **kwargs):
        self._selected_cols = []
        self._lazy = not no_lazy_loading
        if self.splits is not None:
            self.set_split(split)
        if not self._lazy:
            self._load_all()

    def __len__(self):
        return self._length

    @classmethod
    def add_argparse_args(cls, parser, default_split=None):
        if not cls._general_parser_args_added:
            child_parser = parser.add_argument_group(
                "DatasetLoader specific arguments")
            child_parser.add_argument('-p',
                                      '--data_path',
                                      type=str,
                                      required=True,
                                      help="Path to the dataset")
            child_parser.add_argument("--no_lazy_loading",
                                      action="store_true",
                                      help="Disable lazy data loading (some "
                                      "small datasets never use lazy loading)")
            DatasetLoader._general_parser_args_added = True
        if cls.splits is not None and not cls._parser_split_added:
            child_parser.add_argument(
                '-s',
                '--split',
                type=str,
                default="default",
                help="Dataset split to use. The choices depend on the dataset "
                "(see individual dataset groups) (Default is the default "
                "split of the dataset)")
            DatasetLoader._parser_split_added = True
        return parser

    def set_split(self, split_name):
        if self.splits is None:
            raise Exception("This dataset does not have any splits!")
        if split_name not in self.splits:
            if split_name == "default":
                split_name = self.splits[0]
            elif split_name is not None:
                raise KeyError(f"This dataset has no split '{split_name}'!")
        self._cur_split = split_name

    @property
    def trainingset(self):
        return self._datasubset("train")

    @property
    def validationset(self):
        return self._datasubset("valid")

    @property
    def testset(self):
        return self._datasubset("test")

    def _datasubset(self, subset):
        if self._cur_split is None:
            raise KeyError("A split must be selected using '.set_split' "
                           "before accessing a subset")
        elif self._cur_split not in self._splits:
            raise KeyError("This dataset has no split '" + self._cur_split +
                           "'!")
        if subset not in self._splits[self._cur_split]:
            raise KeyError("The split '" + self._cur_split +
                           "' doesn't have a subset " + subset)
        return DataSubset(self, self._cur_split, subset)

    def set_cols(self, *args):
        """
        Sets the data columns to be returned on query.
        Overwrites any previous selection.
        Parameters
        ----------
        strings of data columns to be used.
        """
        for data_key in args:
            if data_key not in self._data_cols:
                raise KeyError("This dataset does not have '" + data_key +
                               "'information.")
        self._selected_cols = list(args)

    def select_col(self, col):
        """
        Add the given column to the list of data returned on query.
        Parameters
        ----------
        col : string
            Name of the data column to be selected
        """
        if col not in self._data_cols:
            raise KeyError("This dataset does not have '" + col +
                           "'information.")
        if col not in self._selected_cols:
            self._selected_cols.append(col)

    def deselect_col(self, col):
        """
        Remove the given column from the list of data returned on query.
        Parameters
        ----------
        col : string
            Name of the data column to be removed from the selection.
        """
        if col in self._selected_cols:
            self._selected_cols.remove(col)

    def has_col(self, col):
        """
        Check whether the dataset has the given type of data.
        Pass in a string key to check its validity for use as key to query
        data.
        Parameters
        ----------
        col : string
            Name of the data column to be checked
        """
        return (col in self._data_cols)

    def __getitem__(self, index):
        """
        Indexing access to the dataset.
        Provides the non-lazy access only. Any dataset to offer lazy access
        must implement the lazy access for any lazy parts manually.
        """
        return {
            data_key: self._data[data_key][index]
            for data_key in self._selected_cols if data_key in self._data
        }

    def iterate(self, split_name=None, split=None, return_tuple=False):
        """
        Iterate over the dataset or a subset of it.
        Parameters
        ----------
        split_name : string, optional
            If given and split is given iterate over the specified data subset
            (if it exists). If None, iterate over the whole dataset.
        split : string, optional
            One of {train, valid, test} If given and split_name is given
            iterate over the specified data subset (if it exists). If None,
            iterate over the whole dataset.
        return_tuple : bool, optional (default is False)
            If True return the data elements as tuples instead of dicts as
            __getitem__does
        """
        if split_name is not None and split is not None:
            index_list = self.get_split(split_name, split)
        else:
            index_list = range(len(self))
        for i in index_list:
            if return_tuple:
                sample = self[i]
                yield tuple(sample[col] for col in self._selected_cols)
            else:
                yield self[i]

    def get_split(self, split_name, split):
        """
        Get indices of elements belonging to a given dataset split.
        Parameters
        ----------
        split_name : string
            Name identifying the dataset split to be returned.
        split_name : string
            One of {train, valid, test}. The datasubset of the given split to
            be returned.
        """
        if split_name not in self._splits:
            raise KeyError("This dataset has no split '" + split_name + "'!")
        if split not in self._splits[split_name]:
            raise KeyError("The split '" + split_name +
                           "' doesn't have a subset " + split)
        return self._splits[split_name][split]

    def _load_all(self):
        """
        Helper for easy non-lazy loading of datasets which do offer lazy
        loading.
        """
        select_cols = self._selected_cols
        self._selected_cols = []
        data = {}
        for col in self._data_cols:
            if col not in self._data.keys():
                self._selected_cols.append(col)
                data[col] = []
        if len(self._selected_cols) > 0:
            for i in range(len(self)):
                sample = self[i]
                for col in self._selected_cols:
                    data[col].append(sample[col])
        for key, val in data.items():
            self._data[key] = val
        self._selected_cols = select_cols


class SkeletonDataset(Dataset):
    """
    Torch dataset class that can be passed into a dataloader.
    Expects a Sequence-type interface (providing __len__ and __getitem__) to
    the data, which returns the data as (keypoints, action_class) tuples, with
    the keypoints in a (persons,frames,landmarks,dimensios) format. Optionally
    can also take a str to a numpy file containing the data in the same format,
    i.e. an array of shape (batch, [keypoints,label],...) where [batch,0]
    contains the keypoint array and [batch,1] the action id.
    Returns samples as tuples (keypoints, action_class) with the keypoints of
    shape either
    (persons, dimensions, frames, landmarks) or (dimensions, frames, landmarks)
    with the latter only occuring when both num_persons == 1 and
    keep_person_dim == False.
    The person dimension is automatically equalised to the selected number by
    truncation/padding with zeros as appropriate.
    Can optionally re-scale the time dimension to a fixed length, using either
    linear interpolation, looping of the sequence or padding with zeros or the
    last frame. Warning: Only interpolation will currently also deal with
    sequences shorter than the target length.
    """
    @staticmethod
    def add_argparse_args(parser: ParserType,
                          default_adjust_len: str = "interpolate",
                          default_target_len: Optional[int] = None,
                          default_num_persons: int = 2) -> ParserType:
        """
        Add skeleton dataset options to argparser.
        Adds command line options for adjusting the length and the person
        dimension of individual sequences.
        Parameters
        ----------
        default_adjust_len : str, optional (default is 'interpolate')
            Sets the default value for the adjust_len parameter.
        default_target_len : int, optional (default is None)
            Sets the default value for the target_len parameter.
        default_num_persons : int, optional (default is 2)
            Sets the default value for the num_persons parameter.
        """
        if isinstance(parser, WithDefaultsWrapper):
            local_parser = parser
        else:
            local_parser = WithDefaultsWrapper(parser)
        local_parser.add_argument(
            '--adjust_len',
            type=str,
            choices=["interpolate", "loop", "pad_zero", "pad_last"],
            default=default_adjust_len,
            help="Adjust the length of individual sequences to a common length"
            " by interpolation, looping the sequence or padding with either "
            "zeros or the last frame.")
        local_parser.add_argument(
            '-l',
            '--target_len',
            type=int,
            default=default_target_len,
            help="Number of frames to scale action sequences to")
        local_parser.add_argument(
            '--num_persons',
            type=int,
            default=default_num_persons,
            help="Number of people to return (extra persons are discarded, "
            "missing persons zero padded)")
        local_parser.add_argument(
            '--keep_person_dim',
            action="store_true",
            help="Only relevevant if num_persons == 1. In that case if set "
            "the keypoint data is returned as a 4D array with a person-"
            "dimension of size 1. If not set the keypoint data is returned as "
            "a 3D array without a person dimension.")

        return parser

    def __init__(self,
                 data: Union[Sequence, str],
                 adjust_len: Optional[str] = None,
                 target_len: Optional[int] = None,
                 num_persons: int = 2,
                 keep_person_dim: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        data : Sequence or str
            A Sequence type object (providing __len__ and __getitem__)
            providing access to the data. Or alternatively a filename of a
            numpy file containing the data in appropriate shape (see class doc
            for a description)
        adjust_len : str, optional (default is None)
            One of ('interpolate', 'loop', 'pad_zero', 'pad_last')
            Optionally adjust the length of each sequence to a fixed number of
            frames by either linear interpoaltion, looping the sequence,
            padding at the end with zeros or padding at the end with the last
            frame. If set target_len must be specified.
        target_len : int, optional (default is None)
            Length to adjust every sample to. Only used if adjust_len is not
            None. Must be specified if adjust_len is not None.
        num_persons : int, optional (default is 2)
            The returned keypoint array is for exactly this number of persons.
            Extra entries in the data are discarded, if fewer skeletons exist
            zeros are aded as padding.
        keep_person_dim : bool, optional (default is False)
            Only relevevant if num_persons == 1. In that case if True the
            keypoint data is returned as a 4D array with a person-dimension of
            size 1. If False the keypoint data is returned as a 3D array
            without a person dimension.
        """
        if isinstance(data, str):
            self._data = np.load(data, allow_pickle=True)
        else:
            self._data = data

        if adjust_len is not None and target_len is None:
            raise ValueError("Target length must be specified when selecting "
                             "to adjust length of samples")
        self._adjust_len = adjust_len
        self._target_len = target_len
        self._num_persons = num_persons
        self._keep_person_dim = keep_person_dim

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Get keypoints and action label for given sequence.
        Adjusts sequence length and person dimension as defined by constructor
        arguments.
        Parameters
        ----------
        index : int
            Index of the dataset sample
        Returns
        -------
        keypoints, action tuple
            Keypoints is numpy array of shape
               (person, coordinates, frame, landmark) or
               (coordinates, frame, landmark)
            action is integer id
        """
        keypoints, action = self._data[index]
        if len(keypoints.shape) == 3:
            # if data has no person dimension temporarily add one
            keypoints = np.expand_dims(keypoints, 0)
        # keypoints = (person, frame, landmark, coordinates)

        # optionally adjust length
        if self._adjust_len is not None:
            keypoints = self._adjust_seq_len(keypoints)

        # reorder for PyTorch channels first convention
        #    -> new order: (person, coordinates, frame, landmark)
        keypoints = keypoints.transpose((0, 3, 1, 2))

        # adjust person dimension if need be
        keypoints = self._adjust_person_dimension(keypoints)

        return np.ascontiguousarray(keypoints, dtype=np.float32), action

    def _adjust_person_dimension(self, data):
        if not self._keep_person_dim and self._num_persons == 1:
            data = data[0]
        elif data.shape[0] < self._num_persons:
            data = np.concatenate(
                (data,
                 np.zeros(
                     (self._num_persons - data.shape[0], ) + data.shape[1:],
                     dtype=data.dtype)),
                axis=0)
        elif data.shape[0] > self._num_persons:
            data = data[:self._num_persons]
        return data

    def _adjust_seq_len(self, data):
        if self._adjust_len == "interpolate":
            # Linearly interpolate the frame dimension
            shape = (data.shape[2], self._target_len)
            rescaled = []
            for i in range(data.shape[0]):
                rescaled.append(
                    cv2.resize(data[i], shape, interpolation=cv2.INTER_LINEAR))
            data = np.stack(rescaled, axis=0)
        elif self._adjust_len == "loop":
            # Loop the frame dimension, repeating the sequence as many times as
            # necessary
            padding_size = self._target_len - data.shape[1]
            full_loops = padding_size // data.shape[1]
            if full_loops > 0:
                padding_size -= full_loops * data.shape[1]
                padding = np.repeat(data, full_loops, axis=1)
                data = np.concatenate((data, padding), axis=1)
            data = np.concatenate((data, data[:, :padding_size]), axis=1)
        elif self._adjust_len.startswith("pad"):
            # Pad the sequence at the end with zeros or the last frame
            padding_size = self._target_len - data.shape[1]
            if self._adjust_len.endswith("zero"):
                padding = np.zeros(
                    (data.shape[0], padding_size) + data.shape[2:],
                    dtype=data.dtype)
            elif self._adjust_len.endswith("last"):
                padding = np.expand_dims(data[:, -1], 1)
                padding = np.repeat(padding, padding_size, axis=1)
            data = np.concatenate((data, padding), axis=1)
        return data

    def get_num_keypoints(self) -> int:
        """
        Determines the number of keypoints from the given data.
        Useful when loading data directly from a file and the info is otw not
        known.
        """
        if len(self._data) == 0:
            raise ValueError("No data to determine number of keypoints.")
        else:
            keypoints, __ = self._data[0]
            # keypoints = (frame, landmark, coordinates) or
            #             (person, frame, landmark, coordinates)
        return keypoints.shape[-2]

    def get_num_actions(self) -> int:
        """
        Determines the number of actions from the given data.
        Useful when loading data directly from a file and the info is otw not
        known. Very inefficient though as it requires going through the data,
        if there is a direct way of knowing the number of actions that should
        be preferred.
        """
        if len(self._data) == 0:
            raise ValueError("No data to determine number of keypoints.")
        else:
            if isinstance(self._data, np.ndarray):
                max_action = np.amax(self._data[:, 1])
            else:
                warn(
                    "get_num_actions from sequence object is very inefficient!"
                )
                max_action = 0
                for __, action in self._data:
                    max_action = max(max_action, action)
            return max_action + 1

    def get_keypoint_dim(self) -> int:
        """
        Determines the dimension of keypoints from the given data.
        Useful when loading data directly from a file and the info is otw not
        known.
        """
        if len(self._data) == 0:
            raise ValueError("No data to determine number of keypoints.")
        else:
            keypoints, __ = self._data[0]
            # keypoints = (frame, landmark, coordinates) or
            #             (person, frame, landmark, coordinates)
        return keypoints.shape[-1]


# these refer to classes in the DatasetLoader package
SUPPORTED_DATASETS = [
    "NTURGBD", "ChaLearn2013", "Skeletics152", "JHMDB", "BerkeleyMHAD"
]
