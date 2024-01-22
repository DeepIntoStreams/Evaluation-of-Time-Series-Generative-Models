from os import path as pt
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from src.evaluations.eval_helper import acf_torch, non_stationary_acf_torch
from src.utils import loader_to_tensor, to_numpy


def plot_summary(fake_dl, real_dl, config, max_lag=None):
    """
    Comparison plots between real and fake paths
    Parameters
    ----------
    fake_dl: torch.utils.data.dataLoader, fake dataset
    real_dl: torch.utils.data.dataLoader, real dataset
    config: configuration file
    max_lag: int, maximum length of the path to be plotted
    """
    x_real, x_fake = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 3, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i:i + 1]
        x_fake_i = x_fake[..., i:i + 1]

        # Plot the marginal density plot
        compare_hists(x_real=to_numpy(x_real_i),
                      x_fake=to_numpy(x_fake_i), ax=axes[i, 0])
        # Plot the marginal log-densit plot
        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(
            x_fake_i), ax=axes[i, 1], log=True)
        # Plot the auto-correlation plot
        compare_acf(x_real=x_real_i, x_fake=x_fake_i,
                    ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))

    plt.savefig(pt.join(config.exp_dir, 'comparison.png'))
    plt.close()

    for i in range(x_real.shape[2]):
        fig = plot_hists_marginals(
            x_real=x_real[..., i:i+1], x_fake=x_fake[..., i:i+1])
        fig.savefig(
            pt.join(config.exp_dir, 'hists_marginals_dim{}.png'.format(i)))
        plt.close()
    plot_samples(real_dl, fake_dl, config)


def plot_hists_marginals(x_real, x_fake):
    """
    Plot the marginal distribution between real and generated data
    Parameters
    ----------
    x_real: torch.tensor, real dataset
    x_fake: torch.tensor, generated dataset
    """
    sns.set()
    n_hists = 10
    n_lags = x_real.shape[1]
    len_interval = n_lags // n_hists
    fig = plt.figure(figsize=(20, 8))

    for i in range(n_hists):
        ax = fig.add_subplot(2, 5, i+1)
        compare_hists(to_numpy(x_real[:, i*len_interval, 0]),
                      to_numpy(x_fake[:, i*len_interval, 0]), ax=ax)
        ax.set_title("Step {}".format(i*len_interval))
    fig.tight_layout()
    return fig


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real = acf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    # acf_real = np.mean(acf_real_list, axis=0)

    acf_fake = acf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    # acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[:, i].shape[0]),
                ub[:, i], lb[:, i],
                color='orange',
                alpha=.3
            )
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()
    return ax


def compare_hists(x_real, x_fake, ax=None, log=False, label=None):
    """ Computes histograms and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = 'Historical ' + label
        label_generated = 'Generated ' + label
    else:
        label_historical = 'Historical'
        label_generated = 'Generated'
    ax.hist(x_real.flatten(), bins=80, alpha=0.6,
            density=True, label=label_historical)[1]
    ax.hist(x_fake.flatten(), bins=80, alpha=0.6,
            density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel('log-pdf')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('pdf')
    return ax


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_samples(real_dl, fake_dl, config, plot_show=False):
    """
    Plot sample paths randomly selected from the generated data
    Parameters
    ----------
    real_dl: torch.dataloader, real dataset
    fake_dl: torch.dataloader, generated dataset
    config: configuration file
    plot_show: boolean, determine where the plot is shown

    Returns
    -------
    The plots will be saved in the experiment directory
    """
    sns.set()
    real_X, fake_X = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    x_real_dim = real_X.shape[-1]

    for i in range(x_real_dim):
        random_indices = torch.randint(0, real_X.shape[0], (250,))
        plt.plot(
            to_numpy(real_X[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
    if plot_show:
        plt.title('real')
        plt.show()
    else:
        plt.savefig(pt.join(config.exp_dir, 'x_real.png'))
    plt.close()

    for i in range(x_real_dim):
        plt.plot(to_numpy(fake_X[:250, :, i]).T, 'C%s' % i, alpha=0.1)
    if plot_show:
        plt.title('fake')
        plt.show()
    else:
        plt.savefig(pt.join(config.exp_dir, 'x_fake.png'))
    plt.close()


def plot_non_stationary_autocorrelation(x1, x2, config, ignore_diagonal=False, plot_show=False):
    """

    Parameters
    ----------
    x1: torch.tensor, auto-correlation tensor of real dataset
    x2: torch.tensor, auto-correlation tensor generated dataset
    config: configuration file
    ignore_diagonal: boolean, it true, the diagonal will be set to 0
    plot_show: boolean, determine where the plot is shown

    Returns
    -------
    The plots will be saved in the experiment directory
    """
    T, _, D = x1.shape

    # Create an array of plots with shape [D, 2]
    fig, axs = plt.subplots(nrows=D, ncols=2, figsize=(12, 12))

    # Loop over each dimension D and plot the heat maps for each input tensor
    for d in range(D):
        # Get the T x T heat map for the d-th dimension of the first and second tensor
        heatmap1 = x1[:, :, d]
        heatmap2 = x2[:, :, d]
        if ignore_diagonal:
            np.fill_diagonal(heatmap1, 0)
            np.fill_diagonal(heatmap2, 0)

        # Add padding to the extent of the heatmaps to create space between the cells

        # Plot the first heat map in the left column
        im1 = axs[d, 0].imshow(heatmap1, cmap='Blues')
        axs[d, 0].set_title(f'Historical Dimension {d}')
        cbar1 = fig.colorbar(im1, ax=axs[d, 0])
        cbar1.mappable.set_clim(vmin=np.min(heatmap1), vmax=np.max(heatmap1))

        # Plot the second heat map in the right column
        im2 = axs[d, 1].imshow(heatmap2, cmap='Blues')
        axs[d, 1].set_title(f'Generated Dimension {d}')
        cbar2 = fig.colorbar(im2, ax=axs[d, 1])
        cbar2.mappable.set_clim(vmin=np.min(heatmap2), vmax=np.max(heatmap2))

    # Adjust the spacing between the plots
    fig.tight_layout()

    # Save the plots
    if plot_show:
        plt.show()
    else:
        plt.savefig(
            pt.join(config.exp_dir, 'non_stationary_autocorrelation.png'))

    plt.close()


def compare_acf_matrix(real_dl, fake_dl, config):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """

    x_real, x_fake = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)

    acf_real = non_stationary_acf_torch(x_real, symmetric=True).cpu().numpy()

    acf_fake = non_stationary_acf_torch(x_fake, symmetric=True).cpu().numpy()

    plot_non_stationary_autocorrelation(acf_real, acf_fake, config)


def tsne_plot(real_dl, fake_dl, config, plot_show=False):
    real_X, fake_X = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(real_X)])
    idx = np.random.permutation(len(real_X))[:anal_sample_no]

  # Data preprocessing
    ori_data = real_X.cpu().numpy()
    generated_data = fake_X.cpu().numpy()

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    no, seq_len, dim = ori_data.shape
    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))
        # Do t-SNE Analysis together
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
    colors = ["red" for i in range(anal_sample_no)] + \
        ["blue" for i in range(anal_sample_no)]
    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    if plot_show:
        plt.show()
    else:
        plt.savefig(pt.join(config.exp_dir, 't-SNE.png'))
    plt.close()