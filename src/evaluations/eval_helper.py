import torch
from torch import nn
import numpy as np
from src.utils import to_numpy
from typing import Tuple
from src.evaluations.augmentations import apply_augmentations, parse_augmentations, Basepoint, Scale
import signatory
import math
from torch.utils.data import DataLoader, TensorDataset
import copy
# from src.baselines.networks.discriminators import Discriminator
from dataclasses import dataclass

def cov_torch(x):
    """Estimates covariance matrix like numpy.cov"""
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


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


def non_stationary_acf_torch(X, symmetric=False):
    """
    Compute the correlation matrix between any two time points of the time series
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    symmetric (bool): whether to return the upper triangular matrix of the full matrix

    Returns
    -------
    Correlation matrix of the shape [T, T, D] where each entry (t_i, t_j, d_i) is the correlation between the d_i-th coordinate of X_{t_i} and X_{t_j}
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    for i in range(D):
        # Compute the correlation between X_{t, d} and X_{t-tau, d}
        if hasattr(torch,'corrcoef'): # version >= torch2.0
            correlations[:, :, i] = torch.corrcoef(X[:, :, i].t())
        else: #TODO: test and fix
            correlations[:, :, i] = torch.from_numpy(np.corrcoef(to_numpy(X[:, :, i]).T))

    if not symmetric:
        # Loop through each time step from lag to T-1
        for t in range(T):
            # Loop through each lag from 1 to lag
            for tau in range(t+1, T):
                correlations[tau, t, :] = 0

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """
    Computes the cross-correlation between feature dimension and time dimension
    Parameters
    ----------
    x
    lags
    dim

    Returns
    -------

    """
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


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

def mean_abs_diff(den1: torch.Tensor,den2: torch.Tensor):
    return torch.mean(torch.abs(den1-den2),0)


def mmd(x,y):
    pass

@dataclass
class ModelSetup:
    model: nn.Module
    optimizer: torch.optim
    criterion: nn.Module
    epochs: int

class TrainValidateTestModel:
    def __init__(self,model=None,optimizer=None,criterion=None,epochs=None,device=None):
        self.model = model 
        self.device = device if device is not None else torch.device('cpu')
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs if epochs is not None else 100

    @staticmethod
    def update_per_epoch(model,optimizer,criterion,
                         dataloader,device,mode,
                         calc_acc
                         ): 
        '''
        mode: train, validate, test
        calc_acc: True for classification, False for regression
        return:
            model, loss, acc
        '''

        model = model.to(device)
        # training
        running_loss = 0
        total = 0
        running_corrects = 0

        if mode == 'train':
            cxt_manager = torch.set_grad_enabled(True)
        elif mode in ['test','validate']:
            cxt_manager = torch.no_grad()
        else:
            raise ValueError('mode must be either train, validate or test')

        # iterate over data
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)            

            with cxt_manager:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if calc_acc:
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        # statistics of the epoch
        loss = running_loss / total
        acc = running_corrects / total if calc_acc else None
        
        # Clean CUDA Memory
        del inputs, outputs, labels
        torch.cuda.empty_cache()
        
        return model, loss, acc

    @staticmethod
    def train_model(model,optimizer,criterion,epochs,device,calc_acc,
                    train_dl, validate_dl=None,
                    valid_condition=None,
                    ):
        """
        Parameters
        ----------
        model: model to be trianed
        optimizer: optimizer of the model's parameters
        criterion: the loss function
        epochs: number of epochs
        device: 'cpu' or 'cuda'
        calc_acc: whether calculate the accuracy for classification tasks
        train_dl: train dataloader
        validate_dl: validation dataloader
        valid_condition: lambda function, controlling the model selection during the validation
        Returns
        -------
        model_setup: class containing the model specifications
        loss: training/validation loss
        acc: accuracy for classification tasks
        """
        best_acc = 0.0
        best_loss = 99
        tranining_loss = None
        validation_loss = None
        training_acc = None
        validation_acc = None

        for epoch in range(epochs):
            # train
            model.train()
            model, tranining_loss, training_acc = __class__.update_per_epoch(model, optimizer, criterion,train_dl,device,mode='train',calc_acc=calc_acc)
            best_model_state_dict = copy.deepcopy(model.state_dict())
            info = f'Epoch {epoch+1}/{epochs} | Loss: {tranining_loss:.4f}'
            info += f' | Acc: {training_acc:.4f}' if calc_acc else ''
            # print(info)

            # validate if condition is not None
            if valid_condition is not None:

                model.eval()
                model, validation_loss, validation_acc = __class__.update_per_epoch(model, None, criterion,validate_dl,device,mode='validate',calc_acc=calc_acc)
                
                if valid_condition(validation_loss,validation_acc,best_acc,best_loss):
                    best_acc = validation_acc
                    best_loss = validation_loss
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    # print(f'Validation | Loss: {loss:.4f} | Acc: {acc:.4f}')

        model.load_state_dict(best_model_state_dict)
        model_setup = ModelSetup(model=model,optimizer=optimizer,criterion=criterion,epochs=epochs)

        loss = validation_loss if validation_loss is not None else tranining_loss
        acc = validation_acc if validation_acc is not None else training_acc

        return model_setup, loss, acc


    @staticmethod
    def test_model(model, criterion,dataloader,device,calc_acc):
        model.eval()
        model.to(device)
        model, loss, acc = __class__.update_per_epoch(model, None, criterion,dataloader,device,mode='test',calc_acc=calc_acc)
        return loss, acc

    def train_val_test_classification(self, train_dl, test_dl, model, train=True, validate=True):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        if train:
            valid_condition = lambda loss,acc,best_acc,best_loss: (acc == best_acc) and (loss <= best_loss) or (acc > best_acc)
            model_setup, _, _ = self.train_model(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                epochs=self.epochs,
                device=self.device,
                calc_acc=True,
                train_dl=train_dl,
                validate_dl=test_dl, # Question: why validate using test_dl?
                valid_condition=valid_condition
                )
        else:
            # TODO: direct testing by loading an existing model
            raise NotImplementedError("The model needs to be trained!")
        test_loss, test_acc = self.test_model(model_setup.model,criterion,test_dl,self.device,calc_acc=True)
        return model_setup, test_loss, test_acc

    def train_val_test_regressor(self, train_dl, test_dl, model, train=True, validate=True):
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        if train:
            valid_condition = lambda loss,acc,best_acc,best_loss: loss <= best_loss
            model_setup, _, _ = self.train_model(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                epochs=self.epochs,
                device=self.device,
                calc_acc=False,
                train_dl=train_dl,
                validate_dl=test_dl,
                valid_condition=valid_condition
                )
        else:
            # TODO: direct testing by loading an existing model
            raise NotImplementedError("The model needs to be trained!")

        test_loss, _ = self.test_model(
            model=model_setup.model,
            criterion=criterion,
            dataloader=test_dl,
            device=self.device,
            calc_acc=False
            )
        return model, test_loss

def create_dl(dl1, dl2, batch_size,cutoff=False):
    x, y = [], []

    if cutoff:
        _, T, C = next(iter(dl1))[0].shape
        T_cutoff = int(T/10)
        for data in dl1:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        for data in dl2:
            x.append(data[0][:, :-T_cutoff])
            y.append(data[0][:, -T_cutoff:].reshape(data[0].shape[0], -1))
        x, y = torch.cat(x), torch.cat(y)
    else:
        for data in dl1:
            x.append(data[0])
            y.append(torch.ones(data[0].shape[0], ))
        for data in dl2:
            x.append(data[0])
            y.append(torch.zeros(data[0].shape[0], ))
        x, y = torch.cat(x), torch.cat(y).long()
    idx = torch.randperm(x.shape[0])
    return DataLoader(TensorDataset(x[idx].view(x.size()), y[idx].view(y.size())), batch_size=batch_size)