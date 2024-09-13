from src.evaluations.augmentations import apply_augmentations, parse_augmentations, Basepoint, Scale
from functools import partial
from typing import Tuple, Optional
from src.utils import to_numpy
import math
# from .trainers.sig_wgan import SigW1Metric

import torch
from torch import nn
import numpy as np
from os import path as pt

import warnings
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
# import signatory
# import ksig
from src.utils import AddTime, set_seed
import signatory
from src.evaluations.metrics import *



def ONND(x_real, x_fake):
    """
    Calculates the Outgoing Nearest Neighbour Distance (ONND) to assess the diversity of the generated data
    Parameters
    ----------
    x_real: torch.tensor, [B, L, D]
    x_fake: torch.tensor, [B, L', D']

    Returns
    -------
    ONND: float
    """
    b1, t1, d1 = x_real.shape
    b2, t2, d2 = x_fake.shape
    assert t1 == t2, "Time length does not agree!"
    assert d1 == d2, "Feature dimension does not agree!"

    # Compute samplewise difference
    x_real_repeated = x_real.repeat_interleave(b2, 0)
    x_fake_repeated = x_fake.repeat([b1, 1, 1])
    samplewise_diff = x_real_repeated - x_fake_repeated
    # Compute samplewise MSE
    MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([b1, -1])
    # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
    ONND = (torch.min(MSE_X_Y, dim=1)[0]).mean()
    return ONND


def INND(x_real, x_fake):
    """
    Calculates the Incoming Nearest Neighbour Distance (INND) to assess the authenticity of the generated data
    Parameters
    ----------
    x_real: torch.tensor, [B, L, D]
    x_fake: torch.tensor, [B, L', D']

    Returns
    -------
    INND: float
    """
    b1, t1, d1 = x_real.shape
    b2, t2, d2 = x_fake.shape
    assert t1 == t2, "Time length does not agree!"
    assert d1 == d2, "Feature dimension does not agree!"

    # Compute samplewise difference
    x_fake_repeated = x_fake.repeat_interleave(b1, 0)
    x_real_repeated = x_real.repeat([b2, 1, 1])
    samplewise_diff = x_real_repeated - x_fake_repeated
    # Compute samplewise MSE
    MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([b2, -1])
    # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
    INND = (torch.min(MSE_X_Y, dim=1)[0]).mean()
    return INND


def ICD(x_fake):
    """
    Calculates the Intra Class Distance (ICD) to detect a potential model collapse
    Parameters
    ----------
    x_fake: torch.tensor, [B, L, D]

    Returns
    -------
    ICD: float
    """
    batch, _, _ = x_fake.shape

    # Compute samplewise difference
    x_fake_repeated_interleave = x_fake.repeat_interleave(batch, 0)
    x_fake_repeated = x_fake.repeat([batch, 1, 1])
    samplewise_diff = x_fake_repeated_interleave - x_fake_repeated
    # Compute samplewise MSE
    MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([batch, -1])
    # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
    ICD = 2 * (MSE_X_Y).sum()
    return ICD / (batch ** 2)

# def Sig_mmd(X, Y, depth,seed=None):
#     """
#     Compute the signature MMD between two distributions
#     Parameters
#     ----------
#     X: torch.tensor, [B, L, D]
#     Y: torch.tensor, [B', L', D']
#     depth: int, signature depth
#
#     Returns
#     -------
#     Sig_MMD between X and Y, torch tensor
#     """
#     # convert torch tensor to numpy
#     N, L, C = X.shape
#     N1, _, C1 = Y.shape
#     X = torch.cat(
#         [torch.zeros((N, 1, C)).to(X.device), X], dim=1)
#     Y = torch.cat(
#         [torch.zeros((N1, 1, C1)).to(X.device), Y], dim=1)
#     X = to_numpy(AddTime(X))
#     Y = to_numpy(AddTime(Y))
#     n_components = 20
#
#     static_kernel = ksig.static.kernels.RBFKernel()
#     # an RBF base kernel for vector-valued data which is lifted to a kernel for sequences
#     static_feat = ksig.static.features.NystroemFeatures(
#         static_kernel, n_components=n_components,random_state=seed)
#     # Nystroem features with an RBF base kernel
#     proj = ksig.projections.CountSketchRandomProjection(
#         n_components=n_components,random_state=seed)
#     # a CountSketch random projection
#
#     lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(
#         n_levels=depth, static_features=static_feat, projection=proj)
#     # sig_kernel = ksig.kernels.SignatureKernel(
#     #   n_levels=depth, static_kernel=static_kernel)
#     # a SignatureKernel object, which works as a callable for computing the signature kernel matrix
#     lr_sig_kernel.fit(X)
#     K_XX = lr_sig_kernel(X)  # K_XX has shape (10, 10)
#     K_XY = lr_sig_kernel(X, Y)
#     K_YY = lr_sig_kernel(Y)
#     m = K_XX.shape[0]
#     diag_X = np.diagonal(K_XX)
#     diag_Y = np.diagonal(K_YY)
#
#     Kt_XX_sums = K_XX.sum(axis=1) - diag_X
#     Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
#     K_XY_sums_0 = K_XY.sum(axis=0)
#
#     Kt_XX_sum = Kt_XX_sums.sum()
#     Kt_YY_sum = Kt_YY_sums.sum()
#     K_XY_sum = K_XY_sums_0.sum()
#     mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
#     mmd2 -= 2 * K_XY_sum / (m * m)
#
#     return torch.tensor(mmd2)

"""
---------------------------- Hypothesis testing related ----------------------------------------
"""


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return torch.tensor(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=False, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    for i in range(n_subsets):
        g = codes_g[choice(len(codes_g), subset_size, replace=False)]
        r = codes_r[choice(len(codes_r), subset_size, replace=False)]
        o = polynomial_mmd(g, r, **kernel_args,
                           var_at_m=m, ret_var=ret_var)
        if ret_var:
            mmds[i], vars[i] = o
        else:
            mmds[i] = o
    return (mmds, vars) if ret_var else torch.tensor(mmds.mean()*1e3)


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)

def FID_score(model, input_real, input_fake):
    """compute the FID score

    Args:
        model (torch model): pretrained rnn model
        input_real (torch.tensor):
        input_fake (torch.tensor):
    """

    device = input_real.device
    linear = model.to(device).linear1
    rnn = model.to(device).rnn
    act_real = linear(rnn(input_real)[
        0][:, -1]).detach().cpu().numpy()
    act_fake = linear(rnn(input_fake)[
        0][:, -1]).detach().cpu().numpy()
    mu_real = np.mean(act_real, axis=0)
    sigma_real = np.cov(act_real, rowvar=False)
    mu_fake = np.mean(act_fake, axis=0)
    sigma_fake = np.cov(act_fake, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def KID_score(model, input_real, input_fake):
    """
    Compute the Kernel Inception Distance (KID) score. The MMD distance between

    Args:
        model (torch model): pretrained rnn model
        input_real (torch.tensor):
        input_fake (torch.tensor):
    """
    device = input_real.device
    linear = model.to(device).linear1
    rnn = model.to(device).rnn
    act_real = linear(rnn(input_real)[
        0][:, -1]).detach().cpu().numpy()
    act_fake = linear(rnn(input_fake)[
        0][:, -1]).detach().cpu().numpy()
    return polynomial_mmd_averages(act_fake, act_real)

