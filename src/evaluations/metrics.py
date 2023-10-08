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
from abc import ABC, abstractmethod
from src.evaluations import eval_helper as eval

'''
Define metrics classes for loss and score computation
Metric List:
- CovarianceMetric
- AutoCorrelationMetric
- CrossCorrelationMetric
- HistogramMetric
- SignatureMetric: SigW1Metric, SigMMDMetric

'''

class Metric(ABC):

    @property
    @abstractmethod
    def name(self):
        pass 

    def measure(self,data, **kwargs):
        pass


class CovarianceMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'CovMetric' 

    def measure(self,data):
        return eval.cov_torch(self.transform(data))

class AutoCorrelationMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'AcfMetric' 

    def measure(self,data,max_lag,stationary,dim=(0, 1),symmetric=False):
        if stationary:
            return eval.acf_torch(self.transform(data),max_lag=max_lag,dim=dim)
        else:
            return eval.non_stationary_acf_torch(self.transform(data),symmetric)
        

class CrossCorrelationMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'CrossCorrMetric' 

    def measure(self,data,lags,dim=(0, 1)):
        return eval.cacf_torch(self.transform(data),lags,dim)
    

class MeanAbsDiffMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'MeanAbsDiffMetric' 

    def measure(self,data):
        x1, x2 = self.transform(data)
        return eval.mean_abs_diff(x1,x2)


class MMDMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'MMDMetric' 

    def measure(self,data):
        x1, x2 = self.transform(data)
        return eval.mmd(x1,x2)


########################## Signature Metric ##########################

class SigExpMetric(Metric):

    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'SigExpMetric'
    
    def measure(self,data: torch.Tensor,depth, augmentations: Optional[Tuple] = (Scale(),), normalise: bool = True):
        assert len(data.shape) == 3, \
            'Path needs to be 3-dimensional. Received %s dimension(s).' % (
                len(data.shape),)
        expected_signature = eval.compute_expected_signature(
            data, depth, augmentations, normalise)
        return expected_signature


class SigMMDMetric(Metric):
        # TODO: make MMD metric
        def __init__(self,transform=lambda x: x):
            self.transform = transform
    
        @property
        def name(self):
            return 'SigMMDMetric'
        
        def measure(self,data: Tuple[torch.Tensor,torch.Tensor], depth, seed = None):
            X, Y = data
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
                static_kernel, n_components=n_components,random_state=seed)
            # Nystroem features with an RBF base kernel
            proj = ksig.projections.CountSketchRandomProjection(
                n_components=n_components,random_state=seed)
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
            mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1)) - 2 * K_XY_sum / (m * m)
            return torch.tensor(mmd2)


class SigW1Metric2(Metric):

    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'SigW1Metric'
    
    def measure(self,data: Tuple[torch.Tensor,torch.Tensor], depth, augmentations: Optional[Tuple] = (Scale(),), normalise: bool = True):
        x_real, x_fake = data
        m = SigExpMetric(self.transform)
        exp_sig_real = m.measure(x_real,depth,augmentations,normalise)
        exp_sig_fake = m.measure(x_fake,depth,augmentations,normalise)
        res = eval.rmse(exp_sig_fake.to(exp_sig_real.device), exp_sig_real)
        return res
  
class SigW1Metric:
    def __init__(self, depth: int, x_real: torch.Tensor, augmentations: Optional[Tuple] = (Scale(),), normalise: bool = True):
        assert len(x_real.shape) == 3, \
            'Path needs to be 3-dimensional. Received %s dimension(s).' % (
                len(x_real.shape),)

        self.augmentations = augmentations
        self.depth = depth
        self.n_lags = x_real.shape[1]

        self.normalise = normalise
        self.expected_signature_mu = eval.compute_expected_signature(
            x_real, depth, augmentations, normalise)

    def __call__(self, x_path_nu: torch.Tensor):
        """ Computes the SigW1 metric."""
        device = x_path_nu.device
        batch_size = x_path_nu.shape[0]
        expected_signature_nu = eval.compute_expected_signature(
            x_path_nu, self.depth, self.augmentations, self.normalise)
        loss = eval.rmse(self.expected_signature_mu.to(
            device), expected_signature_nu)
        return loss