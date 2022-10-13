"""
Simple augmentations to enhance the capability of capturing important features in the first components of the
signature.

Tensors are 3-D arrays corresponding to [batch size, time length, channel dimension]
"""

from dataclasses import dataclass
from typing import List, Tuple

import signatory
import torch


def get_time_vector(size: int, length: int) -> torch.Tensor:
    """
    Size: batch size
    Length: time length
    """
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)
    return x_ll


def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=1)
    x_ll = torch.cat([
        t_rep[:, 0:-2],
        x_rep[:, 1:-1],
        x_rep[:, 2:],
    ], dim=2)
    return x_ll


def sig_normal(sig, normalize=False):
    if normalize == False:
        return sig.mean(0)
    elif normalize == True:
        sig = sig / abs(sig).max(0)[0]
        return sig.mean(0)


def I_visibility_transform(path: torch.Tensor) -> torch.Tensor:
    init_tworows_ = torch.zeros_like(path[:,:1,:])
    init_tworows = torch.cat((init_tworows_, path[:,:1,:]), axis=1)

    a = torch.cat((init_tworows, path), axis=1)

    last_col1 = torch.zeros_like(path[:,:2,:1])
    last_col2 = torch.cat((last_col1, torch.ones_like(path[:,:,:1])), axis=1)

    output = torch.cat((a, last_col2), axis=-1)
    return output

def T_visibility_transform(path: torch.Tensor) -> torch.Tensor:
    """
    The implementation of visibility transformation of segments of path.
     path: dimension (K,a1, a2)
     output path: (K,a1+2,a2+1)
    """

    # adding two rows, the first one repeating the last row of path.

    last_tworows_ = torch.zeros_like(path[:, -1:, :])
    last_tworows = torch.cat((path[:, -1:, :], last_tworows_), axis=1)

    a = torch.cat((path, last_tworows), axis=1)

    # adding a new column, with first path.shape[-1] elements being 1.

    last_col1 = torch.zeros_like(path[:, -2:, :1])
    last_col2 = torch.cat(
        (torch.ones_like(path[:, :, :1]), last_col1), axis=1)

    output = torch.cat((a, last_col2), axis=-1)
    return output


@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1
    dim: int = None

    def apply(self, x: torch.Tensor):
        if self.dim == None:
            return self.scale * x
        else:
            x[...,self.dim] = self.scale * x[...,self.dim]
            return x


@dataclass
class AddTime(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)


@dataclass
class Basepoint(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        basepoint = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        return torch.cat([basepoint, x], dim=1)


@dataclass
class Cumsum(BaseAugmentation):
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return x.cumsum(dim=self.dim)


@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        if self.with_time:
            return lead_lag_transform_with_time(x)
        else:
            return lead_lag_transform(x)

@dataclass
class VisiTrans(BaseAugmentation):
    type: str = "I"

    def apply(self, x: torch.Tensor):
        if self.type == "I":
            return I_visibility_transform(x)
        elif self.type == "T":
            return T_visibility_transform(x)
        else:
            raise ValueError("Unknown type of visibility transform")

@dataclass
class Concat_rtn(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        rtn = x[:,1:,:] - x[:,:-1,:]
        rtn = torch.nn.functional.pad(rtn, (0,0,1,0))
        return torch.cat([x,rtn],2)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y


def get_number_of_channels_after_augmentations(input_dim, augmentations):
    x = torch.zeros(1, 10, input_dim)
    y = apply_augmentations(x, augmentations)
    return y.shape[-1]


AUGMENTATIONS = {'AddTime': AddTime, 'Basepoint': Basepoint, 'CumSum': Cumsum, 'LeadLag': LeadLag,
        'Scale': Scale,  'VisiTrans': VisiTrans, 'Concat_rtn': Concat_rtn}


def parse_augmentations(list_of_dicts):
    augmentations = list()
    for kwargs in list_of_dicts:
        name = kwargs.pop('name')
        augmentations.append(
            AUGMENTATIONS[name](**kwargs)
        )
    return augmentations
