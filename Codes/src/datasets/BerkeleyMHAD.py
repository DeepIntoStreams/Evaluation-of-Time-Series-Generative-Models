

import os
from jcs import JCSAngles
import numpy as np
import torch
from src.datasets.skeleton_dataloader import DatasetLoader, SkeletonDataset
import cv2
import pathlib
from src.datasets.utils import save_data
from tqdm import trange
import random


class BerkeleyMHAD_loader(DatasetLoader):
    """
    BerkeleyMHAD - Berkeley Multimodal Human Action Database
    https://tele-immersion.citris-uc.org/berkeley_mhad
    """
    actions = [
        "Jumping in place", "Jumping jacks",
        "Bending - hands up all the way down", "Punching (boxing",
        "Waving - two hands", "Waving - one hand (right", "Clapping hands",
        "Throwing a ball", "Sit down then stand up", "Sit down", "Stand up"
    ]

    landmarks = [
        "pelvis", "belly", "mid torso", "thorax", "neck", "head top",
        "right shoulder", "right elbow", "right wrist", "left shoulder",
        "left elbow", "left wrist", "right hip", "right knee", "right ankle",
        "right foot", "left hip", "left knee", "left ankle", "left foot"
    ]

    # Order and correspondence of landmarks in the csv file (elements from the
    # csv file in () represent points which don't correspond to a joint but
    # were measured in the mocap system to determin bone rotation. These are
    # dropped using the _landmark_mask):
    # Hips=pelvis, spine=belly, spine1=mid torso, spine2=thorax, Neck=neck,
    # Head=head, (RightShoulder), RightArm=right shoulder, (RightArmRoll),
    # RightForeArm=right elbow, (RightForeArmRoll), RightHand=right wrist,
    # (LeftShoulder), LeftArm=left shoulder, (LeftArmRoll), LeftForeArm=left
    # elbow, (LeftForeArmRoll), LeftHand=left wrist, RightUpLeg=left hip,
    # (RightUpLegRoll), RightLeg=left knee, (RightLegRoll), RightFoot=left
    # ankle, RightToeBase=left foot, LeftUpLeg=right hip, (LeftUpLegRoll),
    # LeftLeg=right knee, (LeftLegRoll), LeftFoot=right ankle,
    # LeftToeBase=right foot
    _landmark_mask = [
        0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 18, 20, 22, 23, 24, 26, 28, 29
    ]

    splits = ["default"]

    @classmethod
    def add_argparse_args(cls, parser, default_split=None):
        super().add_argparse_args(parser, default_split)
        child_parser = parser.add_argument_group(
            "BerkeleyMHAD specific arguments", "Only has the default split.")
        child_parser.add_argument("--subsample",
                                  action="store_true",
                                  help="Subsample data to 30fps")
        return parser

    def __init__(self, data_path, subsample=False, **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        subsample : bool, optional (default is False)
            MoCap data is sampled at 480fps, if subsample is set to True the
            data is subsampled to 30fps
        """
        self._data_cols = [
            "keypoint-filename",
            "keypoints3D",
            "action",
        ]
        self._data = {"keypoint-filename": [], "action": []}

        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in BerkeleyMHAD_loader.splits
        }

        self._subsample = subsample

        self._length = 0
        for subject in range(1, 13):
            for action in range(1, 12):
                for recording in range(1, 6):
                    if subject == 4 and action == 8 and recording == 5:
                        # This sequence is missing in the dataset
                        continue
                    self._data["keypoint-filename"].append(
                        os.path.join(
                            data_path, "Mocap", "SkeletalData", "csv",
                            "skl_s{:02d}_a{:02d}_r{:02d}_pos.csv".format(
                                subject, action, recording)))
                    self._data["action"].append(action - 1)
                    if subject < 8:
                        self._splits["default"]["train"].append(self._length)
                    else:
                        self._splits["default"]["test"].append(self._length)
                    self._length += 1

        super().__init__(**kwargs)

    def load_keypointfile(self, filename):
        """
        Load the keypoints sequence from the given file.
        Parameters
        ----------
        filename : string
            Filename of the file containing a skeleton sequence
        """
        keypoints = []
        if self._subsample:
            counter = 0
        with open(filename, "r") as csv_file:
            csv_file.readline()  # header
            for row in csv_file:
                if self._subsample:
                    counter += 1
                    if counter % 16 != 1:
                        continue
                coords = row.split(",")
                coords = list(map(float, coords[1:]))
                coords = np.array(coords).reshape((-1, 3))
                keypoints.append(coords[self._landmark_mask])
        return np.array(keypoints)

    def __getitem__(self, index):
        """
        Indexing access to the dataset.
        Returns a dictionary of all currently selected data columns of the
        selected item.
        """
        data = super().__getitem__(index)
        # super() provides all non-lazy access, only need to do more for data
        # that hasn't been loaded previously
        if "keypoints3D" in self._selected_cols:
            data["keypoints3D"] = self.load_keypointfile(
                self._data["keypoint-filename"][index])
        return data


class FlatSkeletonDataset(SkeletonDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        keypoints, action = self._data[index]
        if len(keypoints.shape) == 3:
            # If keypoints has no person dimension, temporarily add one
            keypoints = np.expand_dims(keypoints, 0)
        # keypoints = (person, frame, landmark, coordinates)
        keypoints = np.reshape(keypoints, keypoints.shape[:2] + (-1, ))

        # optionally adjust length
        if self._adjust_len is not None:
            if self._adjust_len == "interpolate":
                target_shape = (self._target_len, keypoints.shape[0])
                keypoints = cv2.resize(keypoints,
                                       target_shape,
                                       interpolation=cv2.INTER_LINEAR)
            else:
                keypoints = self._adjust_seq_len(keypoints)

        # Reorder for channels first convention
        #keypoints = np.moveaxis(keypoints, source=-1, destination=-2)
        # (person, angles, frame)

        # adjust person dimension if need be
        keypoints = self._adjust_person_dimension(keypoints)

        return np.ascontiguousarray(keypoints, dtype=np.float32), action


class AngleDataset(SkeletonDataset):
    def __init__(self, landmarks, **kwargs):
        super().__init__(**kwargs)

        #self.movement_angles = MovementAngles(landmarks)
        #self.angle_order = self.movement_angles.get_angle_list()
        # self.angle_order.sort()
        self.jcsangles = JCSAngles(landmarks)
        self.angle_order = self.jcsangles.angle_list

    def __getitem__(self, index):
        data, action = self._data[index]
        if len(data.shape) == 2:
            # If data has no person dimension, temporarily add one
            data = np.expand_dims(data, 0)
        # data = (person, frame, angles)

        # optionally adjust length
        if self._adjust_len is not None:
            if self._adjust_len == "interpolate":
                target_shape = (self._target_len, data.shape[0])
                data = cv2.resize(data,
                                  target_shape,
                                  interpolation=cv2.INTER_LINEAR)
            else:
                if self._adjust_len is not None:
                    data = self._adjust_seq_len(data)

        # Reorder for channels first convention
        #data = np.moveaxis(data, source=-1, destination=-2)
        # (person, angles, frame)

        # adjust person dimension if need be
        data = self._adjust_person_dimension(data)

        return np.ascontiguousarray(data, dtype=np.float32), action

    def convert2angles(self, data):
        angles = []
        if len(data.shape) == 3:
            # If data has no person dimension, add one
            data = np.expand_dims(data, 0)

        # for p in range(data.shape[0]):
        #     person_angles = []
        #     d = data[p]
        #     #d = one_euro_filter(data[p], 30, 0.7, 1)
        #     for frame in range(data.shape[1]):
        #         angle_dict = self.movement_angles.compute_angles(d[frame])
        #         person_angles.append(
        #             list(angle_dict[joint][angle_name]
        #                  for joint, angle_name in self.angle_order))
        #     person_angles = np.array(person_angles)
        #     angles.append(person_angles)
        # return np.array(angles)

        #data = one_euro_filter(data, 30, 0.7, 1)
        angles = self.jcsangles(data)
        return angles

    def get_keypoint_dim(self):
        return len(self.angle_order)


class BerkeleyMHAD:
    def __init__(
        self,
        partition: str,
        data_type: str,
        **kwargs,
    ):
        self.root = pathlib.Path('data')

        self._data_loader = BerkeleyMHAD_loader(
            data_path=self.root/"BerkeleyMHAD/", split='default')
        self._data_loader.set_cols("keypoints3D", "action")

        if data_type == "angles":
            self.DatasetClass = AngleDataset
            data_loc = pathlib.Path(
                'data/BerkeleyMHAD_angles/processed_data')

        elif data_type == "keypoints":
            self.DatasetClass = FlatSkeletonDataset
            data_loc = pathlib.Path(
                'data/BerkeleyMHAD_keypoints/processed_data')
        else:
            raise ValueError("Invalid data type")

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            self.create_datafiles(data_loc=data_loc, data_type=data_type)
        if data_type == "angles":
            if partition == 'train':
                self.X = self.DatasetClass(
                    data=str(data_loc/("training" + ".npy")), landmarks=self._data_loader.landmarks, num_persons=1, adjust_len="interpolate", target_len=50)
            elif partition == 'test':
                self.X = self.DatasetClass(
                    data=str(data_loc/("test" + ".npy")), landmarks=self._data_loader.landmarks, num_persons=1, adjust_len="interpolate", target_len=50)
        elif data_type == 'keypoints':
            if partition == 'train':
                self.X = self.DatasetClass(
                    data=str(data_loc/("training" + ".npy")), num_persons=1, adjust_len="interpolate", target_len=50)
            elif partition == 'test':
                self.X = self.DatasetClass(
                    data=str(data_loc/("test" + ".npy")), num_persons=1, adjust_len="interpolate", target_len=50)

    def create_datafiles(self, data_loc, data_type):
        """
        ----------
        data_files : str
            Path and first part of the file name of the datafiles. The filename
            will automatically be suffixed by the part of the data saved
            (_training or _test) and the filetype (.npy)
        """
        if data_type == 'angles':
            angle_dataset = AngleDataset(data=None,
                                         landmarks=self._data_loader.landmarks)
            for datapart in ("training", "test"):
                filename = data_loc / (datapart + ".npy")
                if not os.path.exists(filename):
                    samples = []
                    subset = eval("self._data_loader." + datapart + "set")
                    for i in trange(len(subset),
                                    desc=f"Creating {datapart}set file"):
                        sample = subset[i]
                        angles = angle_dataset.convert2angles(sample[0])
                        samples.append(
                            np.array([angles, sample[1]], dtype=object))
                    data = np.array(samples, dtype=object)
                    np.save(filename, data)
        elif data_type == 'keypoints':
            for datapart in ("training", "test"):
                filename = data_loc / (datapart + ".npy")
                if not os.path.exists(filename):
                    samples = []
                    subset = eval("self._data_loader." + datapart + "set")
                    for i in trange(len(subset),
                                    desc=f"Creating {datapart}set file"):
                        sample = subset[i]
                        samples.append(
                            np.array([sample[0], sample[1]], dtype=object))
                    data = np.array(samples, dtype=object)
                    np.save(filename, data)


class target_sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, targets, dataset):
        self.mask = self.get_mask(dataset, targets)
        self.dataset = dataset
        self.indices = [i.item() for i in torch.nonzero(self.mask)]

    def __iter__(self):
        return iter(random.sample(self.indices, len(self.indices)))

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_mask(dataset, targets: list) -> torch.Tensor:
        mask = [1 if dataset[i][1]
                in targets else 0 for i in range(len(dataset))]
        mask = torch.tensor(mask)
        return mask


if __name__ == '__main__':
    from src.datasets.BerkeleyMHAD import BerkeleyMHAD, target_sampler
    from torch.utils.data import DataLoader
    import torch
    test_set1 = BerkeleyMHAD(partition='test', data_type='angles')
    test_set2 = BerkeleyMHAD(partition='test', data_type='keypoints')
    sampler = target_sampler(torch.arange(2), test_set1.X)
    test_dl1 = DataLoader(test_set1.X, batch_size=6, sampler=sampler)
    test_dl2 = DataLoader(test_set2.X, batch_size=6, sampler=sampler)
