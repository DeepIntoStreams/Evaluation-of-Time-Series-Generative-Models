from src.evaluations.augmentations import apply_augmentations, parse_augmentations, Basepoint
from functools import partial
from typing import Tuple, Optional
from src.utils import to_numpy
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
from src.utils import AddTime
import signatory


def cov_torch(x, rowvar=False, bias=True, ddof=None, aweights=None):
    # """Estimates covariance matrix like numpy.cov"""
    # reshape x
    # _, T, C = x.shape
    # x = x.reshape(-1, T * C)
    # ensure at least 2D
    # if x.dim() == 1:
    #    x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    # if rowvar and x.shape[0] != 1:
    #    x = x.t()

    # if ddof is None:
    #    if bias == 0:
    #        ddof = 1
    #    else:
    #        ddof = 0

    # w = aweights
    # if w is not None:
    #    if not torch.is_tensor(w):
    #        w = torch.tensor(w, dtype=torch.float)
    #    w_sum = torch.sum(w)
    #    avg = torch.sum(x * (w / w_sum)[:, None], 0)
    # else:
    #    avg = torch.mean(x, 0)

    # Determine the normalization
    # if w is None:
    #    fact = x.shape[0] - ddof
    # elif ddof == 0:
    #    fact = w_sum
    # elif aweights is None:
    #    fact = w_sum - ddof
    # else:
    #    fact = w_sum - ddof * torch.sum(w * w) / w_sum

    # xm = x.sub(avg.expand_as(x))

    # if w is None:
    #    X_T = xm.t()
    # else:
    #    X_T = torch.mm(torch.diag(w), xm).t()

    # c = torch.mm(X_T, xm)
    # c = c / fact

    # return c.squeeze()
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def q_var_torch(x: torch.Tensor):
    """
    :param x: torch.Tensor [B, S, D]
    :return: quadratic variation of x. [B, D]
    """
    return torch.sum(torch.pow(x[:, 1:] - x[:, :-1], 2), 1)


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


def cacf_torch(x, lags: list, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in lags:
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


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
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.acf_real = acf_torch(self.transform(x_real), max_lag, dim=(0, 1))
        self.max_lag = max_lag

    def compute(self, x_fake):
        acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
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


class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(self.transform(x_real))

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(self.transform(x_fake))
        return self.norm_foo(skew_fake - self.skew_real)


class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(self.transform(x_real))

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(self.transform(x_fake))
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)


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
    """compute the KID score

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


def Sig_mmd(X, Y, depth):
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
    def __init__(self, x_real, depth, **kwargs):
        super(Sig_MMD_loss, self).__init__(**kwargs)
        self.x_real = x_real
        self.depth = depth

    def compute(self, x_fake):
        return Sig_mmd(self.x_real, x_fake, self.depth)


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
    def __init__(self, depth: int, x_real: torch.Tensor, augmentations: Optional[Tuple] = (AddTime), normalise: bool = True):
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
    def __init__(self, x_real, **kwargs):
        name = kwargs.pop('name')
        super(SigW1Loss, self).__init__(name=name)
        self.sig_w1_metric = SigW1Metric(x_real=x_real, **kwargs)

    def compute(self, x_fake):
        loss = self.sig_w1_metric(x_fake)
        return loss

# W1 metric


class W1(Loss):
    def __init__(self, D, x_real, **kwargs):
        name = kwargs.pop('name')
        super(W1, self).__init__(name=name)
        self.D = D
        self.D_real = D(x_real).mean()

    def compute(self, x_fake):
        loss = self.D_real-self.D(x_fake).mean()
        return loss


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


def diff(x): return x[:, 1:] - x[:, :-1]


test_metrics = {
    'Sig_mmd': partial(Sig_MMD_loss, name='Sig_mmd', depth=4),
    'SigW1': partial(SigW1Loss, name='SigW1', augmentations=[], normalise=False, depth=4)}


def is_multivariate(x: torch.Tensor):
    """ Check if the path / tensor is multivariate. """
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x: torch.Tensor, **kwargs):
    """ Initialise list of standard test metrics for evaluating the goodness of the generator. """
    if 'model' in kwargs:
        model = kwargs['model']
    test_metrics_list = [test_metrics['Sig_mmd'](x),
                         test_metrics['SigW1'](x)
                         ]
    return test_metrics_list
