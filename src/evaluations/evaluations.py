import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.utils import loader_to_tensor, to_numpy, save_obj
from os import path as pt
import seaborn as sns
from src.evaluations.loss import SigW1Loss, CrossCorrelLoss, HistoLoss, CovLoss, ACFLoss
import numpy as np
import os
import signatory

def _train_classifier(model, train_loader, test_loader, config, epochs=100):
    """
    Train a NN-based classifier to obtain the discriminative score
    Parameters
    ----------
    model: torch.nn.module
    train_loader: torch.utils.data DataLoader: dataset for training
    test_loader: torch.utils.data DataLoader: dataset for testing
    config: configuration file
    epochs: number of epochs for training

    Returns
    -------
    test_acc: model's accuracy in test dataset
    test_loss: model's cross-entropy loss in test dataset
    """
    # Training parameter
    device = config.device
    # clip = config.clip
    # iterate over epochs

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
        # print("Epoch {}/{}".format(epoch + 1, epochs))
        # print("-" * 30)
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
            # print("{} Loss: {:.4f} Acc: {:.4f}".format(
            #     phase, epoch_loss, epoch_acc))

            if phase == "validation" and epoch_acc >= best_acc:
                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

    # print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_acc, test_loss = _test_classifier(
        model, test_loader, config)
    return test_acc, test_loss


def _test_classifier(model, test_loader, config):
    """
    Computes the test metric for trained classifier
    Parameters
    ----------
    model: torch.nn.module, trained model
    test_loader:  torch.utils.data DataLoader: dataset for testing
    config: configuration file

    Returns
    -------
    test_acc: model's accuracy in test dataset
    test_loss: model's cross-entropy loss in test dataset
    """
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
    print("Accuracy of the network on the {} test samples: {}".format(total, (100 * test_acc)))
    return test_acc, test_loss


def _train_regressor(model, train_loader, test_loader, config, epochs=100):
    """
    Training a predictive model to obtain the predictive score
    Parameters
    ----------
    model: torch.nn.module
    train_loader: torch.utils.data DataLoader: dataset for training
    test_loader: torch.utils.data DataLoader: dataset for testing
    config: configuration file
    epochs: number of epochs for training

    Returns
    -------

    """
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

    for epoch in range(epochs):
        # print("Epoch {}/{}".format(epoch + 1, epochs))
        # print("-" * 30)
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
            # print("{} Loss: {:.4f}".format(phase, epoch_loss))

        if phase == "validation" and epoch_loss <= best_loss:

            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Clean CUDA Memory
            del inputs, outputs, labels
            torch.cuda.empty_cache()
    # print("Best Val MSE: {:.4f}".format(best_loss))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_loss = _test_regressor(
        model, test_loader, config)

    return test_loss


def _test_regressor(model, test_loader, config):
    """
    Computes the test metric for trained classifier
    Parameters
    ----------
    model: torch.nn.module, trained model
    test_loader:  torch.utils.data DataLoader: dataset for testing
    config: configuration file

    Returns
    -------
    test_loss: L1 loss between the real and predicted paths by the model in test dataset
    """
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

    test_loss = running_loss / total
    return test_loss


def fake_loader(generator, num_samples, n_lags, batch_size, algo, **kwargs):
    """
    Helper function that transforms the generated data into dataloader, adapted from different generative models
    Parameters
    ----------
    generator: nn.module, trained generative model
    num_samples: int,  number of paths to be generated
    n_lags: int, the length of path to be generated
    batch_size: int, batch size for dataloader
    config: configuration file
    kwargs

    Returns
    Dataload of generated data
    -------

    """
    with torch.no_grad():
        if algo == 'TimeGAN':
            fake_data = generator(batch_size=num_samples,
                                  n_lags=n_lags, device='cpu')
            if 'recovery' in kwargs:
                recovery = kwargs['recovery']
                fake_data = recovery(fake_data)
        elif algo == 'TimeVAE':
            condition = None
            fake_data = generator(num_samples, n_lags,
                                  device='cpu', condition=condition).permute([0, 2, 1])
        else:
            condition = None
            fake_data = generator(num_samples, n_lags,
                                  device='cpu', condition=condition)
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
    Parameters
    ----------
    X: torch.tensor, [N,T,C]
    config: configuration file

    Returns
    -------
    Trained model that minimized the L1 loss between the real and inferred signatures
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
    """
    Evaluation for the synthetic generation, including:
    1) Stylized facts: marginal distribution, cross-correlation, autocorrelation, covariance scores.
    2) Implicit scores: discriminative score, predictive score, predictive_FID.
    3) Rough path scores: SigW1 metric, Signature MMD.
    We compute the mean and std of evaluation scores with 10000 samples and 10 repetitions
    Parameters
    ----------
    real_train_dl: torch.utils.data DataLoader: dataset for training
    real_test_dl: torch.utils.data DataLoader: dataset for testing
    config: configuration file
    kwargs
    Returns
    -------
    Results will be logged in wandb

    """
    sns.set()
    d_scores = []
    p_scores = []
    Sig_MMDs = []
    hist_losses = []
    cross_corrs = []
    cov_losses = []
    acf_losses = []
    sigw1_losses = []

    real_data = torch.cat([loader_to_tensor(real_train_dl),
                          loader_to_tensor(real_test_dl)])
    dim = real_data.shape[-1]

    # TODO: improve (make full eval test_metrics dependent instead of main config)
    if 'algo' in kwargs:
        algo = kwargs['algo']
    else:
        algo = config.algo

    cupy_seed = config.seed if 'seed' in config.keys() else None

    before_update_metrics_config = 'sample_size' in config.keys()
    sample_size = int(config.sample_size) if before_update_metrics_config else 10000
    test_size = int(sample_size * config.test_ratio) if before_update_metrics_config else 2000
    train_size = sample_size - test_size
    batch_size = int(config.batch_size) if before_update_metrics_config else 256

    n = 5
    idx_all = torch.randint(real_data.shape[0], (sample_size*n,)) 

    for i in tqdm(range(n)):
        # take random 10000 samples from real dataset
        # TODO: to update/merge test config later
        idx = idx_all[i*sample_size:(i+1)*sample_size]
        # idx = torch.randint(real_data.shape[0], (sample_size,))
        real_train_dl = DataLoader(TensorDataset(
            real_data[idx[:-test_size]]), batch_size=batch_size)
        real_test_dl = DataLoader(TensorDataset(
            real_data[idx[-test_size:]]), batch_size=batch_size)
        if 'recovery' in kwargs:
            recovery = kwargs['recovery']
            fake_train_dl = fake_loader(generator, num_samples=train_size,
                                        n_lags=config.n_lags, batch_size=batch_size, algo=algo, recovery=recovery)
            fake_test_dl = fake_loader(generator, num_samples=test_size,
                                       n_lags=config.n_lags, batch_size=batch_size, algo=algo, recovery=recovery
                                       )
        else:
            fake_train_dl = fake_loader(generator, num_samples=train_size,
                                        n_lags=config.n_lags, batch_size=batch_size, algo=algo)
            fake_test_dl = fake_loader(generator, num_samples=test_size,
                                       n_lags=config.n_lags, batch_size=batch_size, algo=algo
                                       )

        d_score_mean, d_score_std = compute_discriminative_score(
            real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config, 32, 1, epochs=10, batch_size=128)
        
        d_scores.append(d_score_mean)
        p_score_mean, p_score_std = compute_predictive_score(
            real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config, 32, 2, epochs=10, batch_size=128)
        p_scores.append(p_score_mean)
        real = torch.cat([loader_to_tensor(real_train_dl),
                          loader_to_tensor(real_test_dl)])
        fake = torch.cat([loader_to_tensor(fake_train_dl),
                          loader_to_tensor(fake_test_dl)])
        # predictive_fid = FID_score(fid_model, real, fake)
        # predictive_kid = KID_score(fid_model, real, fake)

        sigw1_losses.append(
            to_numpy(SigW1Loss(x_real=real, depth=2, name='sigw1')(fake)))
        if False:
            sig_mmd = Sig_mmd(real, fake, depth=5,seed=cupy_seed)
            while sig_mmd > 1e3:
                sig_mmd = Sig_mmd(real, fake, depth=5,seed=cupy_seed)
        sig_mmd = to_numpy(SigMMDLoss(x_real=real, depth=5, seed=cupy_seed, name='sigmmd')(fake))
        while sig_mmd > 1e3:
            sig_mmd = to_numpy(SigMMDLoss(x_real=real, depth=5, seed=cupy_seed, name='sigmmd')(fake))

        Sig_MMDs.append(sig_mmd)
        cross_corrs.append(to_numpy(CrossCorrelLoss(
            real, name='cross_correlation')(fake)))
        if config.dataset == 'GBM' or config.dataset == 'ROUGH':
            # Ignore the starting point
            hist_losses.append(
                to_numpy(HistoLoss(real[:, 1:, :], n_bins=50, name='marginal_distribution')(fake[:, 1:, :])))
            # Compute the autocorrelation matrix
            print('compute the autocorrelation matrix')
            acf_losses.append(
                to_numpy(ACFLoss(real, name='acf_loss', stationary=False)(fake)))
        else:
            hist_losses.append(
                to_numpy(HistoLoss(real, n_bins=50, name='marginal_distribution')(fake)))
            acf_losses.append(
                to_numpy(ACFLoss(real, name='acf_loss')(fake)))

        cov_losses.append(to_numpy(CovLoss(real, name='covariance')(fake)))
        # FIDs.append(predictive_fid)
        # KIDs.append(predictive_kid)
    d_mean, d_std = np.array(d_scores).mean(), np.array(d_scores).std()
    p_mean, p_std = np.array(p_scores).mean(), np.array(p_scores).std()
    # fid_mean, fid_std = np.array(FIDs).mean(), np.array(FIDs).std()
    # kid_mean, kid_std = np.array(KIDs).mean(), np.array(KIDs).std()
    sigw1_mean, sigw1_std = np.array(
        sigw1_losses).mean(), np.array(sigw1_losses).std()
    sig_mmd_mean, sig_mmd_std = np.array(
        Sig_MMDs).mean(), np.array(Sig_MMDs).std()
    hist_mean, hist_std = np.array(
        hist_losses).mean(), np.array(hist_losses).std()
    corr_mean, corr_std = np.array(
        cross_corrs).mean(), np.array(cross_corrs).std()
    cov_mean, cov_std = np.array(
        cov_losses).mean(), np.array(cov_losses).std()
    acf_mean, acf_std = np.array(
        acf_losses).mean(), np.array(acf_losses).std()

    # Permutation test
    if 'recovery' in kwargs:
        recovery = kwargs['recovery']
        fake_data = loader_to_tensor(fake_loader(generator, num_samples=int(
            real_data.shape[0]//2), n_lags=config.n_lags, batch_size=128, algo=algo, recovery=recovery))
    else:
        fake_data = loader_to_tensor(fake_loader(generator, num_samples=int(
            real_data.shape[0]//2), n_lags=config.n_lags, batch_size=128, algo=algo))
    power, type1_error = sig_mmd_permutation_test(real_data, fake_data, 5)

    print('discriminative score with mean:', d_mean, 'std:', d_std)
    print('predictive score with mean:', p_mean, 'std:', p_std)
    print('marginal_distribution loss with mean:', hist_mean, 'std:', hist_std)
    print('cross correlation loss with mean:', corr_mean, 'std:', corr_std)
    print('covariance loss with mean:', cov_mean, 'std:', cov_std)
    print('autocorrelation loss with mean:', acf_mean, 'std:', acf_std)
    print('sigw1 with mean:', sigw1_mean, 'std:', sigw1_std)
    print('sig mmd with mean:', sig_mmd_mean, 'std:', sig_mmd_std)
    print('permutation test with power', power, 'type 1 error:', type1_error)

    wandb.run.summary['discriminative_score_mean'] = d_mean
    wandb.run.summary['discriminative_score_std'] = d_std
    wandb.run.summary['predictive_score_mean'] = p_mean
    wandb.run.summary['predictive_score_std'] = p_std
    wandb.run.summary['sigw1_mean'] = sigw1_mean
    wandb.run.summary['sigw1_std'] = sigw1_std
    wandb.run.summary['sig_mmd_mean'] = sig_mmd_mean
    wandb.run.summary['sig_mmd_std'] = sig_mmd_std
    wandb.run.summary['cross_corr_loss_mean'] = corr_mean
    wandb.run.summary['cross_corr_loss_std'] = corr_std
    wandb.run.summary['marginal_distribution_loss_mean'] = hist_mean
    wandb.run.summary['marginal_distribution_loss_std'] = hist_std
    wandb.run.summary['cov_loss_mean'] = cov_mean
    wandb.run.summary['cov_loss_std'] = cov_std
    wandb.run.summary['acf_loss_mean'] = acf_mean
    wandb.run.summary['acf_loss_std'] = acf_std
    wandb.run.summary['permutation_test_power'] = power
    wandb.run.summary['permutation_test_type1_error'] = type1_error