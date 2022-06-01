import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.utils import loader_to_tensor, to_numpy
import matplotlib.pyplot as plt
from os import path as pt
import seaborn as sns
from src.evaluations.test_metrics import kurtosis_torch, skew_torch, cacf_torch
from matplotlib.ticker import MaxNLocator
import numpy as np
"""
def evaluate_generator(experiment_dir, batch_size=1000, device='cpu', foo = lambda x: x):
    generator_config = load_obj(pt.join(experiment_dir, 'generator_config.pkl'))
    generator_state_dict = load_obj(pt.join(experiment_dir, 'generator_state_dict.pt'))
    generator = get_generator(**generator_config)
    generator.load_state_dict(generator_state_dict)

    data_config = load_obj(pt.join(experiment_dir, 'data_config.pkl'))
    x_real = torch.from_numpy(load_obj(pt.join(experiment_dir, 'x_real_test.pkl'))).detach()

    n_lags = data_config['n_lags']

    with torch.no_grad():
        x_fake = generator(batch_size, n_lags, device)
        x_fake = foo(x_fake)

    plot_summary(x_real=x_real, x_fake=x_fake)
    plt.savefig(pt.join(experiment_dir, 'comparison.png'))
    plt.close()

    # compute_discriminative_score(generator, x_real)
    for i in range(x_real.shape[2]):
        fig = plot_hists_marginals(x_real=x_real[...,i:i+1], x_fake=x_fake[...,i:i+1])
        fig.savefig(pt.join(experiment_dir, 'hists_marginals_dim{}.pdf'.format(i)))
        plt.close()
"""


def _train_classifier(
    model, train_loader, test_loader, config, epochs
):
    # Training parameter
    device = config.device
    # clip = config.clip
    dataloader = {'train': train_loader, 'test': test_loader}

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    best_test_acc = 0.0
    best_test_loss = 999
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.CrossEntropyLoss()
    counter = 0
    # wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        #wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "test"]:
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

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    #print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward(retain_graph=True)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "train" and epoch_loss <= best_loss:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    test_acc, test_loss = _test_classifier(
                        model, test_loader, config)
                    best_test_acc, best_test_loss = test_acc, test_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results

                    #test_acc = best_acc
                    counter += 1

        print("best test accuracy:{:.4f}".format(best_test_acc),
              "best test loss:{:.4f}".format(best_test_loss))
        if counter > 50:
            break
    # Report best results
    print("Best test Acc: {:.4f}".format(best_test_acc),
          "Best test Loss: {:.4f}".format(best_test_acc))
    # Load best model weights
    # Return model and histories
    return best_acc, best_loss


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


def fake_loader(generator, num_samples, n_lags, batch_size, config):
    if config.conditional:
        targets = torch.randint(
            0, config.num_classes, (num_samples,))
        condition = one_hot(targets,
                            config.num_classes).float().unsqueeze(1).repeat(1, config.n_lags, 1)
        fake_data = generator(num_samples, n_lags, condition)
    # transform to torch tensor
        tensor_x = torch.Tensor(fake_data[:, :, :-config.num_classes])
        tensor_y = torch.LongTensor(targets)
        return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size)
    else:
        condition = None
        fake_data = generator(num_samples, n_lags, condition)
        tensor_x = torch.Tensor(fake_data)
        return DataLoader(TensorDataset(tensor_x), batch_size=batch_size)


def compute_discriminative_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config,
                                 hidden_size=64, num_layers=3, epochs=100, batch_size=512):

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
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    model = Discriminator(
        train_dl.dataset[0][0].shape[-1], hidden_size, num_layers)
    test_acc, test_loss = _train_classifier(
        model.to(config.device), train_dl, test_dl, config, epochs=epochs)
    return test_loss


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
    #fig.savefig(pt.join(config.exp_dir, 'marginal_comparison.png'))
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
