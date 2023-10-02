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
import ksig
from src.utils import AddTime, set_seed
import signatory


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
        # correlations[:, :, i] = torch.corrcoef(X[:, :, i].t())
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


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()


class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean((0, 1)) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std((0, 1)) - self.std_real)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=cc_diff, **kwargs)
        self.cross_correl_real = cacf_torch(
            self.transform(x_real), max_lag).mean(0)[0]
        self.max_lag = max_lag

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(
            self.transform(x_fake), self.max_lag).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class CovLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CovLoss, self).__init__(norm_foo=cov_diff, **kwargs)
        self.covariance_real = cov_torch(
            self.transform(x_real))

    def compute(self, x_fake):
        covariance_fake = cov_torch(self.transform(x_fake))
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss


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

def Sig_mmd(X, Y, depth):
    """
    Compute the signature MMD between two distributions
    Parameters
    ----------
    X: torch.tensor, [B, L, D]
    Y: torch.tensor, [B', L', D']
    depth: int, signature depth

    Returns
    -------
    Sig_MMD between X and Y, torch tensor
    """
    # convert torch tensor to numpy
    N, L, C = X.shape
    N1, _, C1 = Y.shape
    X = torch.cat(
        [torch.zeros((N, 1, C)).to(X.device), X], dim=1)
    Y = torch.cat(
        [torch.zeros((N1, 1, C1)).to(X.device), Y], dim=1)
    X = to_numpy(AddTime(X))
    Y = to_numpy(AddTime(Y))
    n_components = 20

    static_kernel = ksig.static.kernels.RBFKernel()
    # an RBF base kernel for vector-valued data which is lifted to a kernel for sequences
    static_feat = ksig.static.features.NystroemFeatures(
        static_kernel, n_components=n_components)
    # Nystroem features with an RBF base kernel
    proj = ksig.projections.CountSketchRandomProjection(
        n_components=n_components)
    # a CountSketch random projection

    lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(
        n_levels=depth, static_features=static_feat, projection=proj)
    # sig_kernel = ksig.kernels.SignatureKernel(
    #   n_levels=depth, static_kernel=static_kernel)
    # a SignatureKernel object, which works as a callable for computing the signature kernel matrix
    lr_sig_kernel.fit(X)
    K_XX = lr_sig_kernel(X)  # K_XX has shape (10, 10)
    K_XY = lr_sig_kernel(X, Y)
    K_YY = lr_sig_kernel(Y)
    m = K_XX.shape[0]
    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
    mmd2 -= 2 * K_XY_sum / (m * m)

    return torch.tensor(mmd2)


class Sig_MMD_loss(Loss):
    """
    Signature MMD Loss
    """
    def __init__(self, x_real, depth, **kwargs):
        super(Sig_MMD_loss, self).__init__(**kwargs)
        self.x_real = x_real
        self.depth = depth

    def compute(self, x_fake):
        return Sig_mmd(self.x_real, x_fake, self.depth)


class cross_correlation(Loss):
    def __init__(self, x_real, **kwargs):
        super(cross_correlation).__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake):
        fake_corre = torch.from_numpy(np.corrcoef(
            x_fake.mean(1).permute(1, 0))).float()
        real_corre = torch.from_numpy(np.corrcoef(
            self.x_real.mean(1).permute(1, 0))).float()
        return torch.abs(fake_corre-real_corre)


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


class SigW1Metric:
    def __init__(self, depth: int, x_real: torch.Tensor, augmentations: Optional[Tuple] = (Scale(),), normalise: bool = True):
        assert len(x_real.shape) == 3, \
            'Path needs to be 3-dimensional. Received %s dimension(s).' % (
                len(x_real.shape),)

        self.augmentations = augmentations
        self.depth = depth
        self.n_lags = x_real.shape[1]

        self.normalise = normalise
        self.expected_signature_mu = compute_expected_signature(
            x_real, depth, augmentations, normalise)

    def __call__(self, x_path_nu: torch.Tensor):
        """ Computes the SigW1 metric."""
        device = x_path_nu.device
        batch_size = x_path_nu.shape[0]
        expected_signature_nu = compute_expected_signature(
            x_path_nu, self.depth, self.augmentations, self.normalise)
        loss = rmse(self.expected_signature_mu.to(
            device), expected_signature_nu)
        return loss


class SigW1Loss(Loss):
    def __init__(self, x_real, depth, **kwargs):
        name = kwargs.pop('name')
        super(SigW1Loss, self).__init__(name=name)
        self.sig_w1_metric = SigW1Metric(x_real=x_real, depth=depth, **kwargs)

    def compute(self, x_fake):
        loss = self.sig_w1_metric(x_fake)
        return loss


test_metrics = {
    'Sig_mmd': partial(Sig_MMD_loss, name='Sig_mmd', depth=4),
    'SigW1': partial(SigW1Loss, name='SigW1', augmentations=[], normalise=False, depth=4),
    'marginal_distribution': partial(HistoLoss, n_bins=50, name='marginal_distribution'),
    'cross_correl': partial(CrossCorrelLoss, name='cross_correl'),
    'covariance': partial(CovLoss, name='covariance'),
    'auto_correl': partial(ACFLoss, name='auto_correl')}


def is_multivariate(x: torch.Tensor):
    """ Check if the path / tensor is multivariate. """
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x: torch.Tensor, **kwargs):
    """ Initialise list of standard test metrics for evaluating the goodness of the generator. """
    if 'model' in kwargs:
        model = kwargs['model']
    test_metrics_list = [test_metrics['Sig_mmd'](x),
                         test_metrics['SigW1'](x),
                         test_metrics['marginal_distribution'](x),
                         test_metrics['cross_correl'](x),
                         test_metrics['covariance'](x),
                         test_metrics['auto_correl'](x)
                         ]
    return test_metrics_list


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

class Predictive_FID(Loss):

    def __init__(self, x_real, model, **kwargs):
        super(Predictive_FID, self).__init__(**kwargs)
        self.model = model
        self.x_real = x_real

    def compute(self, x_fake):
        return FID_score(self.model, self.x_real, x_fake)


class Predictive_KID(Loss):
    def __init__(self, x_real, model, **kwargs):
        super(Predictive_KID, self).__init__(**kwargs)
        self.model = model
        self.x_real = x_real

    def compute(self, x_fake):
        return KID_score(self.model, self.x_real, x_fake)


def sig_mmd_permutation_test(X, Y, num_permutation) -> float:
    """two sample permutation test
    Args:
        test_func (function): function inputs: two batch of test samples, output: statistic
        X (torch.tensor): batch of samples (N,C) or (N,T,C)
        Y (torch.tensor): batch of samples (N,C) or (N,T,C)
        num_permutation (int):
    Returns:
        float: test power
    """
    # compute H1 statistics
    # test_func.eval()

    # We first split the data X into two subsets
    idx = torch.randint(X.shape[0], (X.shape[0],))

    X1 = X[idx[-int(X.shape[0]//2):]]
    X = X[idx[:-int(X.shape[0]//2)]]

    with torch.no_grad():

        t0 = Sig_mmd(X, X1, depth=5).cpu().detach().numpy()
        t1 = Sig_mmd(X, Y, depth=5).cpu().detach().numpy()
        print(t1)
        n, m = X.shape[0], Y.shape[0]
        combined = torch.cat([X, Y])

        statistics = []

        for i in range(num_permutation):
            idx1 = torch.randperm(n+m)

            statistics.append(
                Sig_mmd(combined[idx1[:n]], combined[idx1[n:]], depth=5))
            # print(statistics)
        # print(np.array(statistics))
    power = (t1 > torch.tensor(statistics).cpu(
    ).detach().numpy()).sum()/num_permutation
    type1_error = 1 - (t0 > torch.tensor(statistics).cpu(
    ).detach().numpy()).sum()/num_permutation
    return power, type1_error
