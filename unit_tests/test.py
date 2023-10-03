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
from src.utils import loader_to_tensor, set_seed
from src.datasets.dataloader import get_dataset
from src.baselines.networks.TimeVAE import VariationalAutoencoderConvInterpretable
from src.baselines.TimeVAE import TimeVAETrainer
from src.evaluations.test_metrics import get_standard_test_metrics
from src.evaluations.evaluations import full_evaluation
from src.evaluations.summary import full_evaluation_latest
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
        train_dl = torch.load(f"{TestDataSet.config.data_dir}/X_train.pt")
        test_dl = torch.load(f"{TestDataSet.config.data_dir}/X_test.pt")

        self.assertAlmostEqual(loader_to_tensor(test_dl).sum().item(), 398382.46875, delta=TestDataSet.delta)
        self.assertAlmostEqual(loader_to_tensor(train_dl).sum().item(), 1597178.875, delta=TestDataSet.delta)

    def test_dataset_gen(self):
         train_dl, test_dl = get_dataset(TestDataSet.config, num_workers=4, shuffle=False)
         self.assertAlmostEqual(loader_to_tensor(test_dl).sum().item(), 398382.46875, delta=TestDataSet.delta)
         self.assertAlmostEqual(loader_to_tensor(train_dl).sum().item(), 1597178.875, delta=TestDataSet.delta) 

class TestModelVAE(unittest.TestCase):

    # config = test_init(get_test_default_config(model_type='TimeVAE'))
    config = copy.deepcopy(_config_default)
    for k,v in config['TimeVAE'].items():
        config[k] = v

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        config = TestModelVAE.config

        # Fix seed for model initialization
        set_seed(config.seed,device=config.device)

        self.delta = 1e-2
        self.delta_loss = 1

        # Load the same dataset
        self.train_dl = torch.load(f"{config.data_dir}/X_train.pt")
        self.test_dl = torch.load(f"{config.data_dir}/X_test.pt")

        # self.x_real_train = loader_to_tensor(self.train_dl).to(config.device)
        # self.x_real_test = loader_to_tensor(self.test_dl).to(config.device)

        # # config.input_dim = self.x_real_train.shape[-1]

        # self.test_metrics_train = get_standard_test_metrics(self.x_real_train)
        # self.test_metrics_test = get_standard_test_metrics(self.x_real_test)

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

    def test_model_train(self,save=False):
        """
        Test VAE model training
        """
        # Fix seed first before training
        set_seed(TestModelVAE.config.seed,device=TestModelVAE.config.device)

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
        if save:
            save_dir = lambda filename: pt.join(self.__class__.config.data_dir, filename)
            save_obj(self.trainer.G.encoder.state_dict(), save_dir('vae_encoder_state_dict.pt'))
            save_obj(self.trainer.G.decoder.state_dict(), save_dir('vae_decoder_state_dict.pt'))
            save_obj(self.trainer.G, save_dir('vae_model_state_dict.pt'))


    def test_model_eval(self):
        fn = lambda filename: pt.join(__class__.config.data_dir, filename)
        # load pre-trained model and test model
        vae = torch.load(fn('vae_model_state_dict.pt'))
        vae.encoder.load_state_dict(torch.load(fn('vae_encoder_state_dict.pt')), strict=True)
        vae.decoder.load_state_dict(torch.load(fn('vae_decoder_state_dict.pt')), strict=True)
        vae.eval()

        # eval: TODO decompose according to config
        full_evaluation(vae, self.train_dl, self.test_dl,__class__.config)

        # check
        for k in ['discriminative_score_mean',
        'discriminative_score_std',]:
            print(k, wandb.run.summary[k])

        # pass


class TestModelGANs(unittest.TestCase):

    delta = 1e-2
    config = copy.deepcopy(_config_default)
    for k,v in config['TimeGANs'].items():
        config[k] = v

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        config = TestModelGANs.config

        # Fix seed for model initialization
        set_seed(TestModelVAE.config.seed,device=TestModelVAE.config.device)

        # Load the same dataset
        self.train_dl = torch.load(f"{config.data_dir}/X_train.pt")
        self.test_dl = torch.load(f"{config.data_dir}/X_test.pt")

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


    def test_model_train(self,save=False):
        """
        Test TimeGAN model training
        """
        # Fix seed first before training
        set_seed(TestModelVAE.config.seed,device=TestModelVAE.config.device)


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

    config = copy.deepcopy(_config_default)
    delta = 1e-2

    # TODO decompose according to config
    ref_val_map = {
        'discriminative_score_mean': 0.067,
        'discriminative_score_std': 0.10556,
        'predictive_score_mean': 0.91824,
        'predictive_score_std': 0.02692,
        # 'sigw1_mean': 0.96064,
        # 'sigw1_std': 0.03814,
        # 'sig_mmd_mean':5.23808,
        # 'sig_mmd_std': 1.57949,
        'cross_corr_loss_mean':0.24941,
        'cross_corr_loss_std': 0.01789,
        'marginal_distribution_loss_mean': 0.2832,
        'marginal_distribution_loss_std': 0.00889,
        'cov_loss_mean': 0.08304,
        'cov_loss_std': 0.00373,
        # 'acf_loss_mean': np.nan,
        # 'acf_loss_std': np.nan,
        'permutation_test_power': 1.0,
        # 'permutation_test_type1_error': 0.2
    }

    rename_map = {
        'cross_corr_loss_mean': 'cross_corr_mean',
        'cross_corr_loss_std': 'cross_corr_std',
        'marginal_distribution_loss_mean':'hist_loss_mean',
        'marginal_distribution_loss_std':'hist_loss_std',
    }

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        set_seed(__class__.config.seed,device=__class__.config.device)

        self.train_dl = torch.load(pt.join(__class__.config.data_dir, 'X_train.pt'))
        self.test_dl = torch.load(pt.join(__class__.config.data_dir, 'X_test.pt'))


    def test_metric_vae(self):
        # TODO: use vae as an example, will regroup metrics later
        config = __class__.config
        fn = lambda filename: pt.join(config.data_dir, filename)

        # load pre-trained model
        vae = torch.load(fn('vae_model_state_dict.pt'))
        vae.encoder.load_state_dict(torch.load(fn('vae_encoder_state_dict.pt')), strict=True)
        vae.decoder.load_state_dict(torch.load(fn('vae_decoder_state_dict.pt')), strict=True)
        vae.eval()

        # eval: TODO decompose according to config
        set_seed(config.seed,device=config.device)

        use_original = 0
        if use_original:
            full_evaluation(vae, self.train_dl, self.test_dl, config, algo='TimeVAE')
            for k,val in self.ref_val_map.items():
                # print(k,wandb.run.summary[k])
                self.assertAlmostEqual(wandb.run.summary[k], val, delta=self.__class__.delta)
        else:
            summary = full_evaluation_latest(vae, self.train_dl, self.test_dl, config, algo='TimeVAE')       
            ref_name_map = {v: k for k, v in self.rename_map.items()}
            for k,val in summary.items():
                    kref = ref_name_map.get(k,k)
                    self.assertAlmostEqual(summary[k], self.ref_val_map[kref], delta=self.__class__.delta)

        # full_evaluation(vae, self.train_dl, self.test_dl, config, algo='TimeVAE')       

        # check
        # for k,val in summary.items():
        #     # print(k,wandb.run.summary[k])
        #     self.assertAlmostEqual(wandb.run.summary[k], val, delta=self.__class__.delta)







if __name__ == '__main__':
    unittest.main()


# python -m unittest discover -v