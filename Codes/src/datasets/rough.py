from tqdm import tqdm
import numpy as np
from fbm import MBM
import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split


def get_rBergomi_paths(hurst=0.25, size=20000, n_lags=50, maturity=1, xi=0.1, eta=0.5):
    r"""
    Paths of Rough stochastic volatility model for an asset price process S_t of the form

    dS_t = \sqrt(V_t) S_t dZ_t
    V_t := \xi * exp(\eta * W_t^H - 0.5*\eta^2*t^{2H})

    where W_t^H denotes the Riemann-Liouville fBM given by

    W_t^H := \int_0^t K(t-s) dW_t,  K(r) := \sqrt{2H} r^{H-1/2}

    with W_t,Z_t correlated brownian motions (I'm actually considering \rho=0)

    Parameters
    ----------
    hurst: float,
    size: int
        size of the dataset
    n_lags: int
        Number of timesteps in the path
    maturity: float
        Final time. Should be a value in [0,1]
    xi: float
    eta: float

    Returns
    -------
    dataset: np.array
        array of shape (size, n_lags, 2)

    """
    assert hurst < 0.5, "hurst parameter should be < 0.5"

    dataset = np.zeros((size, n_lags, 2))
    print('Generate 20000 path samples of rough stochastics volatility')
    for j in tqdm(range(size), total=size):
        # we generate v process
        m = MBM(n=n_lags-1, hurst=lambda t: hurst,
                length=maturity, method='riemannliouville')
        fbm = m.mbm()  # fractional Brownian motion
        times = m.times()
        V = xi * np.exp(eta * fbm - 0.5 * eta**2 * times**(2*hurst))

        # we generate price process
        h = times[1:] - times[:-1]  # time increments
        brownian_increments = np.random.randn(h.shape[0]) * np.sqrt(h)

        log_S = np.zeros_like(V)
        # Ito formula to get SDE for  d log(S_t). We assume S_0 = 1
        log_S[1:] = (-0.5 * V[:-1]*h + np.sqrt(V[:-1])
                     * brownian_increments).cumsum()
        S = np.exp(log_S)
        dataset[j] = np.stack([S, V], 1)
    return dataset


class Rough_S(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        **kwargs,
    ):
        n_lags = n_lags

        data_loc = pathlib.Path(
            'data/ROUGH/processed_data_{}'.format(n_lags))

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = get_rBergomi_paths(n_lags=n_lags)
            x_real = torch.from_numpy(x_real).float()
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(Rough_S, self).__init__(X)

    @staticmethod
    def load_data(data_loc, partition):

        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
        elif partition == "test":
            X = tensors["test_X"]
        else:
            raise NotImplementedError(
                "the set {} is not implemented.".format(set))

        return X
