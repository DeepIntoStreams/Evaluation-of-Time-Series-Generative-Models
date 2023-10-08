from random import shuffle
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.datasets.GBM import get_GBM_paths
from src.datasets.AR1 import AROne
from src.evaluations.test_metrics import Sig_mmd
from src.evaluations.evaluations import compute_discriminative_score, compute_predictive_score, sig_fid_model
from torch.utils.data import DataLoader, TensorDataset
from src.evaluations.test_metrics import Predictive_FID, Predictive_KID
from src.utils import to_numpy


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

    def permutation_test(self, test_func, num_perm, sample_size):
        with torch.no_grad():
            X = self.subsample(self.X, sample_size)
            Y = self.subsample(self.Y, sample_size)
            X = X.to(self.config.device)
            Y = Y.to(self.config.device)

            # print(t1)
            n, m = X.shape[0], Y.shape[0]
            combined = torch.cat([X, Y])
            H0_stats = []
            H1_stats = []

            for i in range(num_perm):
                idx = torch.randperm(n+m)
                H0_stats.append(
                    test_func(combined[idx[:n]], combined[idx[n:]]).cpu().detach().numpy())
                H1_stats.append(test_func(self.subsample(
                    self.X, sample_size).to(self.config.device), self.subsample(self.Y, sample_size).to(self.config.device)).cpu().detach().numpy())
            Q_a = np.quantile(np.array(H0_stats), q=0.95)
            Q_b = np.quantile(np.array(H1_stats), q=0.05)

            # print(statistics)
            # print(np.array(statistics))
            power = 1 - (Q_a > np.array(H1_stats)).sum()/num_perm
            type1_error = (Q_b < np.array(H0_stats)).sum()/num_perm
        return power, type1_error


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
        X = get_GBM_paths(size=50000, n_lags=30, dim=5, drift=0.02, scale=0.1)
        Y = get_GBM_paths(size=50000, n_lags=30, dim=5, drift=0.04, scale=0.1)
    df = Compare_test_metrics(X, Y, config).run_montontic_test(
        num_run=3, distubance_level=4, sample_size=10000)
    df.to_csv('numerical_results/test_metrics_test_' +
              config.dataset + '_complexnet.csv')
