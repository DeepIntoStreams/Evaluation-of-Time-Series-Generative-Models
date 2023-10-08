import torch
from torch import nn
import numpy as np
from src.utils import to_numpy
from typing import Tuple
from src.evaluations.augmentations import apply_augmentations, parse_augmentations, Basepoint, Scale
import signatory
import math

def cov_torch(x):
    """Estimates covariance matrix like numpy.cov"""
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """
    Compute the correlation matrix between any two time points of the time series
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    symmetric (bool): whether to return the upper triangular matrix of the full matrix

    Returns
    -------
    Correlation matrix of the shape [T, T, D] where each entry (t_i, t_j, d_i) is the correlation between the d_i-th coordinate of X_{t_i} and X_{t_j}
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    for i in range(D):
        # Compute the correlation between X_{t, d} and X_{t-tau, d}
        if hasattr(torch,'corrcoef'): # version >= torch2.0
            correlations[:, :, i] = torch.corrcoef(X[:, :, i].t())
        else: #TODO: test and fix
            correlations[:, :, i] = torch.from_numpy(np.corrcoef(to_numpy(X[:, :, i]).T))

    if not symmetric:
        # Loop through each time step from lag to T-1
        for t in range(T):
            # Loop through each lag from 1 to lag
            for tau in range(t+1, T):
                correlations[tau, t, :] = 0

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """
    Computes the cross-correlation between feature dimension and time dimension
    Parameters
    ----------
    x
    lags
    dim

    Returns
    -------

    """
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
    x_path_augmented = apply_augmentations(x_path, augmentations)
    expected_signature = signatory.signature(
        x_path_augmented, depth=depth).mean(0)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            expected_signature[count:count + dim**(
                i+1)] = expected_signature[count:count + dim**(i+1)] * math.factorial(i+1)
            count = count + dim**(i+1)
    return expected_signature


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()

def mean_abs_diff(den1: torch.Tensor,den2: torch.Tensor):
    return torch.mean(torch.abs(den1-den2),0)


def mmd(x,y):
    pass