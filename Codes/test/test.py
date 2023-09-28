import unittest
import ml_collections
import copy
import wandb
import yaml
import os
import sys
from src.baselines.models import get_trainer
from test.test_utils import *
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



class TestDataSet(unittest.TestCase):
    '''
    add test for real data generation, including
    note: we may use saved data for further testing
    '''
    config = get_test_default_config()
    config = test_init(config)
    delta = 1e-2

    def test_dataset(self):
        """
        Test that it can sum a list of integers
        """
        train_dl, test_dl = get_dataset(TestDataSet.config, num_workers=4, shuffle=False)
        self.assertAlmostEqual(loader_to_tensor(test_dl).sum().item(), 398382.46875, delta=TestDataSet.delta)
        self.assertAlmostEqual(loader_to_tensor(train_dl).sum().item(), 1597178.875, delta=TestDataSet.delta)


class TestModelVAE(unittest.TestCase):
    
    config = test_init(get_test_default_config(model_type='TimeVAE'))

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        config = TestModelVAE.config

        self.delta = 1e-2
        self.delta_loss = 1
        self.model = VariationalAutoencoderConvInterpretable(
            hidden_layer_sizes=config.hidden_layer_sizes,
            trend_poly=config.trend_poly,
            num_gen_seas=config.num_gen_seas,
            custom_seas=config.custom_seas,
            use_scaler=config.use_scaler,
            use_residual_conn=config.use_residual_conn,
            n_lags=config.n_lags,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            reconstruction_wt=config.reconstruction_wt
            )
        
        if True:
            self.train_dl, self.test_dl = get_dataset(config, num_workers=4, shuffle=False)
            # torch.save(self.train_dl, config.data_dir_X_train)
            # torch.save(self.test_dl, config.data_dir_X_test)
        else:
            self.train_dl = torch.load(config.data_dir_X_train)
            self.test_dl = torch.load(config.data_dir_X_test)
        
        self.x_real_train = loader_to_tensor(self.train_dl).to(config.device)
        self.x_real_test = loader_to_tensor(self.test_dl).to(config.device)
        
        self.test_metrics_train = get_standard_test_metrics(self.x_real_train)
        self.test_metrics_test = get_standard_test_metrics(self.x_real_test)
        
        self.trainer = get_trainer(config, self.train_dl, self.test_dl)

    def test_model_pre_train(self):
        param_dict = {
            'encoder.encoder.conv_0.weight' : 76.1750,
            'encoder.encoder_mu.weight' : 924.4887,
            'encoder.encoder_log_var.weight' : 921.4078,
            'decoder.level_model.model.0.weight' : 8.1777,
            'decoder.residual_model.model.last_conv.weight' : 76.8107
        }     
        for k,v in param_dict.items():
            # print(k, self.trainer.G.state_dict()[k].abs().sum().item(), v)
            self.assertAlmostEqual(self.trainer.G.state_dict()[k].abs().sum().item(), v, delta=self.delta)

    def test_model_train(self):

        self.trainer.fit(device=TestModelVAE.config.device)

        # fitted model
        param_dict = {
            'encoder.encoder.conv_0.weight' : 77.7656,
            'encoder.encoder_mu.weight' : 933.6230,
            'encoder.encoder_log_var.weight' : 937.5732,
            'decoder.level_model.model.0.weight' : 7.8881,
            'decoder.residual_model.model.last_conv.weight' : 74.8925
        }     
        for k,v in param_dict.items():
            # print(k, self.trainer.G.state_dict()[k].abs().sum().item(), v)
            self.assertAlmostEqual(self.trainer.G.state_dict()[k].abs().sum().item(), v, delta=self.delta)

        # loss
        final_loss = 26.9895
        self.assertAlmostEqual(wandb.run.summary['G_loss'], final_loss, delta=self.delta_loss)

        # # Test whether the loss has decreased
        # self.assertLess(final_loss, initial_loss)

        
    def test_model_eval(self):
        # load pre-trained model and test model
        pass


class TestModelGANs(unittest.TestCase):
    
    config = get_test_default_config()
    delta = 1e-2

    def test_model_train(self):
        pass

    def test_model_generator(self):
        pass

    def test_model_eval(self):
        pass


class TestMetrics(unittest.TestCase):
    
    config = get_test_default_config()
    delta = 1e-2

    def test_metrics(self):
        pass





if __name__ == '__main__':
    unittest.main()


# python -m unittest discover -v