from random import shuffle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from src.evaluations.test_metrics import Sig_mmd
from functools import partial
import torch.nn as nn
from src.evaluations.evaluations import compute_discriminative_score, compute_predictive_score, sig_fid_model
from torch.utils.data import DataLoader, TensorDataset
import signatory
from src.evaluations.evaluations import _train_regressor
from src.evaluations.test_metrics import Predictive_FID, Predictive_KID
from src.utils import to_numpy


def sig_mmd_permutation_test(X, X1, Y, num_permutation) -> float:
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


def get_gbm(size, n_lags, d=1, drift=0., scale=0.1, h=1):
    x_real = torch.ones(size, n_lags, d)
    x_real[:, 1:, :] = torch.exp(
        (drift-scale**2/2)*h + (scale*np.sqrt(h)*torch.randn(size, n_lags-1, d)))
    x_real = x_real.cumprod(1)
    return x_real


class AROne:
    '''
    :param D: dimension of x
    :param T: sequence length
    :param phi: parameters for AR model
    :param s: parameter that controls the magnitude of covariance matrix
    '''

    def __init__(self, D, T, phi, s, burn=10):
        self.D = D
        self.T = T
        self.phi = phi
        self.Sig = np.eye(D) * (1 - s) + s
        self.chol = np.linalg.cholesky(self.Sig)
        self.burn = burn

    def batch(self, N):
        x0 = np.random.randn(N, self.D)
        x = np.zeros((self.T + self.burn, N, self.D))
        x[0, :, :] = x0
        for i in range(1, self.T + self.burn):
            x[i, ...] = self.phi * x[i - 1] + \
                np.random.randn(N, self.D) @ self.chol.T

        x = x[-self.T:, :, :]
        x = np.swapaxes(x, 0, 1)
        return x.astype("float32")


class Compare_test_metrics:

    def __init__(self, X, Y, config):
        self.X = X
        self.Y = Y
        self.config = config

    @staticmethod
    def subsample(X, sample_size):
        if sample_size > X.shape[0]:
            raise ValueError('required samples size is larger than data size')
        idx = torch.randperm(X.shape[0])
        return X[idx[:sample_size]]

    @staticmethod
    def create_monotonic_dataset(X, X1, Y, i):
        #  replace Y by X up to dimension i, as i increases the dicrepency is smaller
        if i == 0:
            Y = X1  # when disturbance is 0, we use data from the same distribution as X, but not X
        else:
            Y[..., :-i] = X1[..., :-i]
        return X, Y

    def run_montontic_test(self, num_run: int, distubance_level: int, sample_size):
        d_scores = []
        p_scores = []
        Sig_MMDs = []
        disturbance = []
        sig_fids = []
        sig_kids = []
        X = self.subsample(self.X, sample_size)
        fid_model = sig_fid_model(X, self.config)
        for i in tqdm(range(distubance_level+1)):
            for j in range(num_run):
                X1 = self.subsample(self.X, sample_size)
                Y = self.subsample(self.Y, sample_size)
                X, Y = self.create_monotonic_dataset(X, X1, Y, i)
                X_train_dl = DataLoader(TensorDataset(
                    X[:-2000]), batch_size=128)
                X_test_dl = DataLoader(TensorDataset(
                    X[-2000:]), batch_size=128)
                Y_train_dl = DataLoader(TensorDataset(
                    Y[:-2000]), batch_size=128)
                Y_test_dl = DataLoader(TensorDataset(
                    Y[-2000:]), batch_size=128)

                d_score_mean, _ = compute_discriminative_score(
                    X_train_dl, X_test_dl, Y_train_dl, Y_test_dl, self.config, self.config.dscore_hidden_size,
                    num_layers=self.config.dscore_num_layers, epochs=self.config.dscore_epochs, batch_size=128)
                d_scores.append(d_score_mean)
                p_score_mean, _ = compute_predictive_score(
                    X_train_dl, X_test_dl, Y_train_dl, Y_test_dl, self.config, self.config.pscore_hidden_size,
                    self.config.pscore_num_layers, epochs=self.config.pscore_epochs, batch_size=128)
                p_scores.append(p_score_mean)
                sig_fid = Predictive_FID(
                    X, model=fid_model, name='Predictive_FID')(Y)
                sig_fids.append(to_numpy(sig_fid))
                sig_kid = Predictive_KID(
                    X, model=fid_model, name='Predictive_KID')(Y)
                sig_kids.append(to_numpy(sig_kid))

                sig_mmd = Sig_mmd(X, Y, depth=4)
                Sig_MMDs.append(to_numpy(sig_mmd))
                disturbance.append(i)
        return pd.DataFrame({'sig_mmd': Sig_MMDs, 'signature fid': sig_fids, 'signature kid': sig_kids, 'predictive scores': p_scores, 'discriminative score': d_scores, 'disturbance': disturbance})


if __name__ == '__main__':
    import ml_collections
    import yaml
    import os
    config_dir = 'configs/' + 'test_metrics.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if config.dataset == 'AR':
        X = torch.FloatTensor(
            AROne(D=5, T=30, phi=np.linspace(0.1, 0.5, 5), s=0.5).batch(50000))
        Y = torch.FloatTensor(
            AROne(D=5, T=30, phi=np.linspace(0.6, 1, 5), s=0.5).batch(50000))
    if config.dataset == 'GBM':
        X = get_gbm(50000, 30, 5, drift=0.02, scale=0.1)
        Y = get_gbm(50000, 30, 5, drift=0.04, scale=0.1)
    df = Compare_test_metrics(X, Y, config).run_montontic_test(
        num_run=3, distubance_level=4, sample_size=10000)
    df.to_csv('numerical_results/test_metrics_test_' +
              config.dataset + '_complexnet.csv')
