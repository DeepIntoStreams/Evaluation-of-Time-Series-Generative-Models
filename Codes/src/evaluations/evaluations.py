import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.utils import loader_to_tensor, to_numpy, save_obj
import matplotlib.pyplot as plt
from os import path as pt
import seaborn as sns
from src.evaluations.test_metrics import Predictive_KID, Sig_mmd, kurtosis_torch, skew_torch, cacf_torch, FID_score, KID_score, CrossCorrelLoss, HistoLoss
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.manifold import TSNE
import warnings
import os
import signatory


def _train_classifier(model, train_loader, test_loader, config, epochs=100):
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    dataloader = {'train': train_loader, 'validation': test_loader}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    criterion = torch.nn.CrossEntropyLoss()
    # wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                if config.dataset == 'MNIST':
                    inputs = inputs.squeeze(1).permute(0, 2, 1)
                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)
                # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_acc, test_loss = _test_classifier(
        model, test_loader, config)
    return test_acc, test_loss


def _test_classifier(model, test_loader, config):
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0
    running_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            if config.dataset == 'MNIST':

                inputs = inputs.squeeze(1).permute(0, 2, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    test_loss = running_loss / total
    print(
        "Accuracy of the network on the {} test samples: {}".format(
            total, (100 * test_acc)
        )
    )
    return test_acc, test_loss


def _train_regressor(
    model, train_loader, test_loader, config, epochs=100
):
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999
    dataloader = {'train': train_loader, 'validation': test_loader}
    criterion = torch.nn.L1Loss()
    # wandb.watch(model, criterion, log="all", log_freq=1)
    # wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            total = 0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(True):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    # print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
            epoch_loss = running_loss / total
            print("{} Loss: {:.4f}".format(
                phase, epoch_loss))
        if phase == "validation" and epoch_loss <= best_loss:

            # Updates to the weights will not happen if the accuracy is equal but loss does not diminish

            best_loss = epoch_loss

            best_model_wts = copy.deepcopy(model.state_dict())

            # Log best results so far and the weights of the model.

            # Clean CUDA Memory
            del inputs, outputs, labels
            torch.cuda.empty_cache()
    print("Best Val MSE: {:.4f}".format(best_loss))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    epoch_loss = _test_regressor(
        model, test_loader, config)

    return best_loss


def _test_regressor(model, test_loader, config):
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    total = 0
    running_loss = 0
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            total += labels.size(0)

    # Print results
    test_loss = running_loss / total

    return test_loss


def fake_loader(generator, num_samples, n_lags, batch_size, config, **kwargs):
    if 'recovery' in kwargs:
        recovery = kwargs['recovery']
    with torch.no_grad():
        if config.algo == 'TimeGAN':
            fake_data = generator(batch_size=num_samples,
                                  n_lags=n_lags, device='cpu')
            fake_data = recovery(fake_data)
        elif config.algo == 'TimeVAE':
            condition = None
            fake_data = generator(num_samples, n_lags,
                                  device='cpu', condition=condition).permute([0, 2, 1])
            print(fake_data.shape)

        else:
            condition = None
            fake_data = generator(num_samples, n_lags,
                                  device='cpu', condition=condition)
            print(fake_data.shape)
        tensor_x = torch.Tensor(fake_data)
    return DataLoader(TensorDataset(tensor_x), batch_size=batch_size)


def compute_discriminative_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config,
                                 hidden_size=64, num_layers=2, epochs=30, batch_size=512):

    def create_dl(real_dl, fake_dl, batch_size):
        train_x, train_y = [], []
        for data in real_dl:
            train_x.append(data[0])
            train_y.append(torch.ones(data[0].shape[0], ))
        for data in fake_dl:
            train_x.append(data[0])
            train_y.append(torch.zeros(data[0].shape[0], ))
        x, y = torch.cat(train_x), torch.cat(train_y).long()
        idx = torch.randperm(x.shape[0])

        return DataLoader(TensorDataset(x[idx].view(x.size()), y[idx].view(y.size())), batch_size=batch_size)
    train_dl = create_dl(real_train_dl, fake_train_dl, batch_size)
    test_dl = create_dl(real_test_dl, fake_test_dl, batch_size)

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size=2):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers,
                              hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    test_acc_list = []
    for i in range(1):
        model = Discriminator(
            train_dl.dataset[0][0].shape[-1], hidden_size, num_layers)

        test_acc, test_loss = _train_classifier(
            model.to(config.device), train_dl, test_dl, config, epochs=epochs)
        test_acc_list.append(test_acc)
    mean_acc = np.mean(np.array(test_acc_list))
    std_acc = np.std(np.array(test_acc_list))
    return abs(mean_acc-0.5), std_acc


def plot_samples(real_dl, fake_dl, config):
    sns.set()
    real_X, fake_X = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    x_real_dim = real_X.shape[-1]
    for i in range(x_real_dim):
        plt.plot(to_numpy(fake_X[:250, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(pt.join(config.exp_dir, 'x_fake.png'))
    plt.close()

    for i in range(x_real_dim):
        random_indices = torch.randint(0, real_X.shape[0], (250,))
        plt.plot(
            to_numpy(real_X[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(pt.join(config.exp_dir, 'x_real.png'))
    plt.close()


def plot_samples1(real_dl, fake_dl, config):
    sns.set()
    real_X, fake_X = loader_to_tensor(real_dl), loader_to_tensor(fake_dl)
    x_real_dim = real_X.shape[-1]
    for i in range(x_real_dim):
        random_indices = torch.randint(0, real_X.shape[0], (100,))
        plt.plot(to_numpy(fake_X[:100, :, i]).T, 'C%s' % i, alpha=0.1)
        plt.plot(
            to_numpy(real_X[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
        plt.savefig(pt.join(config.exp_dir, 'sample_plot{}.png'.format(i)))
        plt.close()


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


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


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
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


def plot_hists_marginals(x_real, x_fake):
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
    # fig.savefig(pt.join(config.exp_dir, 'marginal_comparison.png'))
    # plt.close(fig)
    return fig


def plot_summary(fake_dl, real_dl, config, max_lag=None):
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

        compare_hists(x_real=to_numpy(x_real_i),
                      x_fake=to_numpy(x_fake_i), ax=axes[i, 0])

        def text_box(x, height, title):
            textstr = '\n'.join((
                r'%s' % (title,),
                # t'abs_metric=%.2f' % abs_metric
                r'$s=%.2f$' % (skew_torch(x).item(),),
                r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].text(
                0.05, height, textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )

        text_box(x_real_i, 0.95, 'Historical')
        text_box(x_fake_i, 0.70, 'Generated')

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(
            x_fake_i), ax=axes[i, 1], log=True)
        # compare_acf(x_real=x_real_i, x_fake=x_fake_i,
        #           ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))
    plt.savefig(pt.join(config.exp_dir, 'comparison.png'))
    plt.close()

    for i in range(x_real.shape[2]):
        fig = plot_hists_marginals(
            x_real=x_real[..., i:i+1], x_fake=x_fake[..., i:i+1])
        fig.savefig(
            pt.join(config.exp_dir, 'hists_marginals_dim{}.pdf'.format(i)))
        plt.close()
    plot_samples(real_dl, fake_dl, config)


def compute_classfication_score(real_train_dl, fake_train_dl, config,
                                hidden_size=64, num_layers=3, epochs=100):
    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(Discriminator, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)
    model = Discriminator(
        real_train_dl.dataset[0][0].shape[-1], hidden_size, num_layers, out_size=config.num_classes)
    TFTR_acc = _train_classifier(
        model.to(config.device), fake_train_dl, real_train_dl, config, epochs=epochs)
    TRTF_acc = _train_classifier(
        model.to(config.device), real_train_dl, fake_train_dl, config, epochs=epochs)
    return TFTR_acc, TRTF_acc


def compute_predictive_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config,
                             hidden_size=64, num_layers=3, epochs=100, batch_size=128):
    def create_dl(train_dl, test_dl, batch_size):
        x, y = [], []
        _, T, C = next(iter(train_dl))[0].shape

        T_cutoff = int(T/10)
        for data in train_dl:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        for data in test_dl:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        x, y = torch.cat(x), torch.cat(y),
        idx = torch.randperm(x.shape[0])
        dl = DataLoader(TensorDataset(x[idx].view(
            x.size()), y[idx].view(y.size())), batch_size=batch_size)

        return dl
    train_dl = create_dl(fake_train_dl, fake_test_dl, batch_size)
    test_dl = create_dl(real_train_dl, real_test_dl, batch_size)

    class predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(predictor, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    test_loss_list = []
    for i in range(1):
        model = predictor(
            train_dl.dataset[0][0].shape[-1], hidden_size, num_layers, out_size=train_dl.dataset[0][1].shape[-1])
        test_loss = _train_regressor(
            model.to(config.device), train_dl, test_dl, config, epochs=epochs)
        test_loss_list.append(test_loss)
    mean_loss = np.mean(np.array(test_loss_list))
    std_loss = np.std(np.array(test_loss_list))
    return mean_loss, std_loss


def visualization(real_dl, fake_dl, config):
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
    plt.savefig(pt.join(config.exp_dir, 't-SNE.png'))
    plt.close()


# def FID_score()


def train_predictive_FID_model(real_train_dl, real_test_dl, config,
                               hidden_size=64, num_layers=3, epochs=100, batch_size=64):
    # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py

    class predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(predictor, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear1 = nn.Linear(hidden_size, 256)
            self.linear2 = nn.Linear(256, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            x = self.linear1(x)
            return self.linear2(x)

    real_train_dl = DataLoader(
        real_train_dl.dataset, batch_size=batch_size, shuffle=True)
    real_test_dl = DataLoader(real_test_dl.dataset,
                              batch_size=batch_size, shuffle=True)
    model_dir = './numerical_results/{dataset}/evaluate_model/'.format(
        dataset=config.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    else:
        pass

    if os.path.exists(pt.join(model_dir, 'fid_predictive_model.pt')):
        if config.dataset == 'Pendigt' or config.dataset == 'SpeechCommands' or config.dataset == 'MNIST':
            model = predictor(
                config.input_dim, hidden_size, num_layers, out_size=config.num_classes)
            model.load_state_dict(torch.load(
                pt.join(model_dir, 'fid_predictive_model.pt')), strict=True)
        else:
            model = predictor(
                real_train_dl.dataset[0][0].shape[-1], hidden_size, num_layers, out_size=real_train_dl.dataset[0][1].shape[-1])
            model.load_state_dict(torch.load(
                pt.join(model_dir, 'fid_predictive_model.pt')), strict=True)
    else:
        if config.dataset == 'Pendigt' or config.dataset == 'SpeechCommands' or config.dataset == 'MNIST':

            # train_sampler = pen_target_sampler(
            #   torch.arange(config.num_classes), real_train_dl.dataset.tensors[1])
            # test_sampler = pen_target_sampler(
            #   torch.arange(config.num_classes), real_test_dl.dataset.tensors[1])
            real_train_dl = DataLoader(
                real_train_dl.dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            real_test_dl = DataLoader(
                real_test_dl.dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            model = predictor(
                config.input_dim, hidden_size, num_layers, out_size=config.num_classes)

            test_acc, test_loss = _train_classifier(
                model.to(config.device), real_train_dl, real_test_dl, config, epochs=epochs)
            print('predictive FID test accuracy:', test_acc)
            save_obj(model.state_dict(), pt.join(
                model_dir, 'fid_predictive_model.pt'))
            torch.save(model.state_dict(),
                       pt.join(wandb.run.dir, 'fid_predictive_model.pt'))

        else:
            model = predictor(
                real_train_dl.dataset[0][0].shape[-1], hidden_size, num_layers, out_size=real_train_dl.dataset[0][1].shape[-1])
            test_loss = _train_regressor(
                model.to(config.device), real_train_dl, real_test_dl, config, epochs=epochs)

            save_obj(model.state_dict(), pt.join(
                model_dir, 'fid_predictive_model.pt'))
            torch.save(model.state_dict(),
                       pt.join(wandb.run.dir, 'fid_predictive_model.pt'))
            print('predictive FID test mse:', test_loss)
    return model


def sig_fid_model(X: torch.tensor, config):
    """

    Args:
        X (torch.tensor): time series tensor N,T,C
    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    Y = signatory.signature(X, 3)
    N = X.shape[0]
    real_train_dl = DataLoader(TensorDataset(
        X[:int(N*0.8)], Y[:int(N*0.8)]), batch_size=128, shuffle=True)
    real_test_dl = DataLoader(TensorDataset(X[int(N*0.8):], Y[int(N*0.8):]),
                              batch_size=128, shuffle=False)

    class predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(predictor, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear1 = nn.Linear(hidden_size, 512)
            self.linear2 = nn.Linear(512, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            x = self.linear1(x)
            return self.linear2(x)
    model = predictor(
        real_train_dl.dataset[0][0].shape[-1], 64, 2, out_size=Y.shape[-1])
    test_loss = _train_regressor(
        model.to(config.device), real_train_dl, real_test_dl, config, epochs=200)
    print('predictive FID test mse:', test_loss)

    return model


def full_evaluation(generator, real_train_dl, real_test_dl, config, **kwargs):
    """ evaluation for the synthetic generation, including.
        discriminative score, predictive score, predictive_FID, predictive_KID
        We compute the mean and std of evaluation scores 
        with 10000 samples and 10 repetitions  
    Args:
        generator (_type_): torch.model
        real_X (_type_): torch.tensor
    """
    d_scores = []
    p_scores = []
    Sig_MMDs = []
    hist_losses = []
    cross_corrs = []
    real_data = torch.cat([loader_to_tensor(real_train_dl),
                          loader_to_tensor(real_test_dl)])
    dim = real_data.shape[-1]

    for i in tqdm(range(5)):
        # take random 10000 samples from real dataset
        idx = torch.randint(real_data.shape[0], (10000,))
        real_train_dl = DataLoader(TensorDataset(
            real_data[idx[:-2000]]), batch_size=128)
        real_test_dl = DataLoader(TensorDataset(
            real_data[idx[-2000:]]), batch_size=128)
        if 'recovery' in kwargs:
            recovery = kwargs['recovery']
            fake_train_dl = fake_loader(generator, num_samples=8000,
                                        n_lags=config.n_lags, batch_size=128, config=config, recovery=recovery)
            fake_test_dl = fake_loader(generator, num_samples=2000,
                                       n_lags=config.n_lags, batch_size=128, config=config, recovery=recovery
                                       )
        else:
            fake_train_dl = fake_loader(generator, num_samples=8000,
                                        n_lags=config.n_lags, batch_size=128, config=config)
            fake_test_dl = fake_loader(generator, num_samples=2000,
                                       n_lags=config.n_lags, batch_size=128, config=config
                                       )

        d_score_mean, d_score_std = compute_discriminative_score(
            real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config, int(dim/2), 1, epochs=30, batch_size=128)
        d_scores.append(d_score_mean)
        p_score_mean, p_score_std = compute_predictive_score(
            real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config, 32, 2, epochs=50, batch_size=128)
        p_scores.append(p_score_mean)
        real = torch.cat([loader_to_tensor(real_train_dl),
                          loader_to_tensor(real_test_dl)])
        fake = torch.cat([loader_to_tensor(fake_train_dl),
                          loader_to_tensor(fake_test_dl)])
        #predictive_fid = FID_score(fid_model, real, fake)
        #predictive_kid = KID_score(fid_model, real, fake)
        sig_mmd = Sig_mmd(real, fake, depth=5)
        Sig_MMDs.append(sig_mmd)
        cross_corrs.append(to_numpy(CrossCorrelLoss(real)(fake)))
        hist_losses.append(to_numpy(HistoLoss(real)(fake)))
        # FIDs.append(predictive_fid)
        # KIDs.append(predictive_kid)
    d_mean, d_std = np.array(d_scores).mean(), np.array(d_scores).std()
    p_mean, p_std = np.array(p_scores).mean(), np.array(p_scores).std()
    #fid_mean, fid_std = np.array(FIDs).mean(), np.array(FIDs).std()
    #kid_mean, kid_std = np.array(KIDs).mean(), np.array(KIDs).std()
    sig_mmd_mean, sig_mmd_std = np.array(
        Sig_MMDs).mean(), np.array(Sig_MMDs).std()
    hist_mean, hist_std = np.array(
        hist_losses).mean(), np.array(hist_losses).std()
    corr_mean, corr_std = np.array(
        cross_corrs).mean(), np.array(cross_corrs).std()

    print('discriminative score with mean:', d_mean, 'std:', d_std)
    print('predictive score with mean:', p_mean, 'std:', p_std)
    print('histogram loss with mean:', hist_mean, 'std:', hist_std)
    print('cross correlation loss with mean:', corr_mean, 'std:', corr_std)
    print('sig mmd with mean:', sig_mmd_mean, 'std:', sig_mmd_std)
    wandb.run.summary['discriminative_score_mean'] = d_mean
    wandb.run.summary['discriminative_score_std'] = d_std

    wandb.run.summary['predictive_score_mean'] = p_mean
    wandb.run.summary['predictive_score_std'] = p_std
    wandb.run.summary['sig_mmd_mean'] = sig_mmd_mean
    wandb.run.summary['sig_mmd_std'] = sig_mmd_std
    wandb.run.summary['cross_corr_loss_mean'] = corr_mean
    wandb.run.summary['cross_corr_loss_std'] = corr_std

    wandb.run.summary['hist_loss_mean'] = hist_mean
    wandb.run.summary['hist_loss_std'] = hist_std
