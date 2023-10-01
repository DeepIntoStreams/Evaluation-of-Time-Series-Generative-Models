from contextlib import AbstractContextManager
from typing import Any
import unittest
import ml_collections
import copy
import numpy as np
import torch
import wandb
import yaml
import os
import sys
from src.baselines.models import get_trainer
from unit_tests.test_utils import *
from src.utils import loader_to_tensor
from src.datasets.dataloader import get_dataset
from src.baselines.networks.TimeVAE import VariationalAutoencoderConvInterpretable
from src.baselines.TimeVAE import TimeVAETrainer
from src.evaluations.test_metrics import get_standard_test_metrics
'''
We need to decide what we want to test. The list below is a good starting point:
fix seed
- data processing
- model training
- model evaluation
- model saving
'''

_config_default = test_init()

class TestDataSet(unittest.TestCase):
    '''
    add test for real data generation, including
    note: we may use saved data for further testing
    '''
    # config = get_test_default_config()
    # config = test_init(config)
    config = copy.deepcopy(_config_default)
    delta = 1e-2

    def test_dataset(self):
        """
        Test that it can sum a list of integers
        """
        # print(TestDataSet.config)

        train_dl = torch.load("unit_tests/X_train.pt")
        test_dl = torch.load("unit_tests/X_test.pt")

        self.assertAlmostEqual(loader_to_tensor(test_dl).sum().item(), 398382.46875, delta=TestDataSet.delta)
        self.assertAlmostEqual(loader_to_tensor(train_dl).sum().item(), 1597179.0, delta=TestDataSet.delta)


class TestModelVAE(unittest.TestCase):

    # config = test_init(get_test_default_config(model_type='TimeVAE'))
    config = copy.deepcopy(_config_default)
    for k,v in config['TimeVAE'].items():
        config[k] = v

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        config = TestModelVAE.config

        # Fix seed for model initialization
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.delta = 1e-2
        self.delta_loss = 1

        # Load the same dataset
        self.train_dl = torch.load(config.data_dir_X_train)
        self.test_dl = torch.load(config.data_dir_X_test)

        self.x_real_train = loader_to_tensor(self.train_dl).to(config.device)
        self.x_real_test = loader_to_tensor(self.test_dl).to(config.device)

        config.input_dim = self.x_real_train.shape[-1]

        self.test_metrics_train = get_standard_test_metrics(self.x_real_train)
        self.test_metrics_test = get_standard_test_metrics(self.x_real_test)

        self.trainer = get_trainer(config, self.train_dl, self.test_dl)
        torch.backends.cudnn.benchmark = False

    def test_model_pre_train(self):
        """
        Test VAE model initialization
        """
        # Check
        param_dict = {
            'encoder.encoder.conv_0.weight' : 78.8649,
            'encoder.encoder_mu.weight' : 918.5593,
            'encoder.encoder_log_var.weight' : 924.9228,
            'decoder.level_model.model.0.weight' : 7.4125,
            'decoder.residual_model.model.last_conv.weight' : 79.2322
        }
        for k,v in param_dict.items():
            # print(k, "True: ", self.trainer.G.state_dict()[k].abs().sum().item(), "Test: ", v)
            self.assertAlmostEqual(self.trainer.G.state_dict()[k].abs().sum().item(), v, delta=self.delta)

    def test_model_train(self):
        """
        Test VAE model training
        """
        # Fix seed first before training
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.trainer.fit(device=TestModelVAE.config.device)

        # fitted model
        param_dict = {
            'encoder.encoder.conv_0.weight': 78.7601,
            'encoder.encoder_mu.weight': 931.9981,
            'encoder.encoder_log_var.weight': 937.4478,
            'decoder.level_model.model.0.weight': 7.3538,
            'decoder.residual_model.model.last_conv.weight': 78.2277
        }

        for k,v in param_dict.items():
            # print(k, "True: ", self.trainer.G.state_dict()[k].abs().sum().item(), "Test: ", v)
            self.assertAlmostEqual(self.trainer.G.state_dict()[k].abs().sum().item(), v, delta=self.delta)

        # loss
        final_loss = 28.8234
        self.assertAlmostEqual(wandb.run.summary['G_loss'], final_loss, delta=self.delta_loss)

        # # Test whether the loss has decreased
        # self.assertLess(final_loss, initial_loss)


    def test_model_eval(self):
        # load pre-trained model and test model
        pass


class TestModelGANs(unittest.TestCase):

    delta = 1e-2
    config = copy.deepcopy(_config_default)
    for k,v in config['TimeGANs'].items():
        config[k] = v

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        config = TestModelGANs.config

        # Fix seed for model initialization
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Load the same dataset
        self.train_dl = torch.load(config.data_dir_X_train)
        self.test_dl = torch.load(config.data_dir_X_test)

        self.x_real_train = loader_to_tensor(self.train_dl).to(config.device)
        self.x_real_test = loader_to_tensor(self.test_dl).to(config.device)

        config.input_dim = self.x_real_train.shape[-1]

        self.trainer = get_trainer(config, self.train_dl, self.test_dl)
        torch.backends.cudnn.benchmark = False

    def test_model_pre_train(self):
        """
        Test TimeGAN model initialization
        """
        param_dict = {
            'rnn.weight_ih_l0': 82.0279,
            'rnn.bias_ih_l1': 15.9273,
            'linear.weight': 26.1998,
        }
        for k,v in param_dict.items():
            # print(k, self.trainer.G.state_dict()[k].abs().sum().item(), v)
            self.assertAlmostEqual(self.trainer.G.state_dict()[k].abs().sum().item(), v, delta=self.delta)


    def test_model_train(self):
        """
        Test TimeGAN model training
        """
        # Fix seed first before training
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.trainer.fit(device=TestModelGANs.config.device)
        param_dict = {
            'rnn.weight_ih_l0': 83.3145,
            'rnn.bias_ih_l1': 15.9317,
            'linear.weight': 26.4506,
        }
        for k,v in param_dict.items():
            # print(k, "True: ", self.trainer.G.state_dict()[k].abs().sum().item(), "False: ", v)
            self.assertAlmostEqual(self.trainer.G.state_dict()[k].abs().sum().item(), v, delta=self.delta)


class TestMetrics(unittest.TestCase):

    config = get_test_default_config()
    delta = 1e-2

    def test_metrics(self):
        pass





if __name__ == '__main__':
    unittest.main()


# python -m unittest discover -v