import numpy as np
from os import path as pt

import warnings
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
# import signatory
# import ksig
from src.utils import AddTime
import signatory
from src.evaluations.metrics import *
from src.evaluations.eval_helper import *
from src.evaluations.test_metrics import * #TODO: remove as all metrics will be included in metrics.py


def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x, seed=None):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.seed = seed

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()
    
    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        self.metric = AutoCorrelationMetric(self.transform)
        self.acf_calc = lambda x: self.metric.measure(x, self.max_lag, stationary,dim=(0, 1),symmetric=False)
        self.acf_real = self.acf_calc(x_real)

    def compute(self, x_fake):
        acf_fake = self.acf_calc(x_fake)
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
        self.lags = max_lag
        self.metric = CrossCorrelationMetric(self.transform)
        self.cross_correl_real = self.metric.measure(x_real,self.lags).mean(0)[0]
        self.max_lag = max_lag

    def compute(self, x_fake):
        cross_correl_fake = self.metric.measure(x_fake,lags=self.lags).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


# unused
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
        self.metric = CovarianceMetric(self.transform)
        self.covariance_real = self.metric.measure(x_real)

    def compute(self, x_fake):
        covariance_fake = self.metric.measure(x_fake)
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss


# class SigMMDLoss(Loss):
#     """
#     Signature MMD Loss
#     """
#     def __init__(self, x_real, depth, **kwargs):
#         super(SigMMDLoss, self).__init__(**kwargs)
#         self.x_real = x_real
#         self.depth = depth
#         self.seed = kwargs.get('seed',None)
#
#     def compute(self, x_fake):
#         m = SigMMDMetric(self.transform)
#         return m.measure((self.x_real, x_fake), self.depth, seed=self.seed)
    
class SigW1Loss(Loss):
    def __init__(self, x_real, depth, normalise, **kwargs):
        name = kwargs.pop('name')
        super(SigW1Loss, self).__init__(name=name)
        self.sig_w1_metric = SigW1Metric(x_real=x_real, depth=depth, normalise=normalise, **kwargs)

    def compute(self, x_fake):
        loss = self.sig_w1_metric(x_fake)
        return loss


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


class VARLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(VARLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='var')

    def compute(self, x_fake):
        loss = list()
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='var')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                loss.append(abs_metric)
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

class ESLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(ESLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='es')

    def compute(self, x_fake):
        loss = list()
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='es')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                loss.append(abs_metric)
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

def tail_metric(x, alpha, statistic):
    res = list()
    for i in range(x.shape[2]):
        tmp_res = list()
        # Exclude the initial point
        for t in range(x.shape[1]):
            x_ti = x[:, t, i].reshape(-1, 1)
            sorted_arr, _ = torch.sort(x_ti)
            var_alpha_index = int(alpha * len(sorted_arr))
            var_alpha = sorted_arr[var_alpha_index]
            if statistic == "es":
                es_values = sorted_arr[:var_alpha_index + 1]
                es_alpha = es_values.mean()
                tmp_res.append(es_alpha)
            else:
                tmp_res.append(var_alpha)
        res.append(tmp_res)
    return res

#################### Standard Metrics ####################

test_metrics = {
    # 'Sig_mmd': partial(SigMMDLoss, name='Sig_mmd', depth=4),
    'SigW1': partial(SigW1Loss, name='SigW1', augmentations=[], normalise=False, depth=4),
    'marginal_distribution': partial(HistoLoss, n_bins=50, name='marginal_distribution'),
    'cross_correl': partial(CrossCorrelLoss, name='cross_correl'),
    'covariance': partial(CovLoss, name='covariance'),
    'auto_correl': partial(ACFLoss, name='auto_correl')
    }


def get_standard_test_metrics(x: torch.Tensor, **kwargs):
    """ Initialise list of standard test metrics for evaluating the goodness of the generator. """
    if 'model' in kwargs:
        model = kwargs['model']
    test_metrics_list = [
        # test_metrics['Sig_mmd'](x),
                         test_metrics['SigW1'](x),
                         test_metrics['marginal_distribution'](x),
                         test_metrics['cross_correl'](x),
                         test_metrics['covariance'](x),
                         test_metrics['auto_correl'](x)
                         ]
    return test_metrics_list


def permutation_test(test_func_arg_tuple, X, Y, n_permutation) -> float:
    ''' two sample permutation test general 
    test_func (function): 
        - function inputs: two batch of test samples, 
        - output: statistic
    '''
    test_func, kwargs = test_func_arg_tuple

    idx = torch.randint(X.shape[0], (X.shape[0],))
    X1 = X[idx[-int(X.shape[0]//2):]]
    X = X[idx[:-int(X.shape[0]//2)]]
    with torch.no_grad():
        t0 = to_numpy(test_func(X, X1,**kwargs))
        t1 = to_numpy(test_func(X, Y,**kwargs))

        n, m = X.shape[0], Y.shape[0]
        combined = torch.cat([X, Y])

        statistics = []
        for i in range(n_permutation):
            idx1 = torch.randperm(n+m)
            stat = test_func(combined[idx1[:n]], combined[idx1[n:]],**kwargs)
            statistics.append(stat)

    power = (t1 > to_numpy(torch.tensor(statistics))).sum()/n_permutation
    type1_error = 1 - (t0 > to_numpy(torch.tensor(statistics))).sum()/n_permutation

    return power, type1_error

# def sig_mmd_permutation_test(X, Y, n_permutation) -> float:
#     test_func_arg_tuple = (Sig_mmd,{'depth':5})
#     return permutation_test(
#         test_func_arg_tuple, X, Y, n_permutation)

