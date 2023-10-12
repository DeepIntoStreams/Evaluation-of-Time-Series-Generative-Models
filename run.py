"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import ml_collections
import copy
import wandb
import yaml
import os

from os import path as pt
import numpy as np
from src.evaluations.evaluations import fake_loader, full_evaluation
from src.evaluations.plot import plot_summary, compare_acf_matrix
import torch
from src.utils import get_experiment_dir, set_seed, convert_config_to_dict
from torch import nn
import argparse


def main(config):
    """
    Main interface, provides model the model training with target datasets and final assessment of trained model
    Parameters
    ----------
    config: configuration file
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    set_seed(config.seed)

    # Flat the config file by reading the model config
    for k,v in config.Model[config.algo].items():
        config[k] = v

    del config.Model

    # initialize weight and bias
    # Place here your API key.
    # setup own api key in the config
    os.environ["WANDB_API_KEY"] = config.WANDB.wandb_api
    tags = [
        config.algo,
        config.dataset,
    ]

    config = convert_config_to_dict(copy.deepcopy(dict(config)))

    wandb.init(
        project='Generative_model_evaluation',
        config=config,
        entity="deepintostreams",
        tags=tags,
        group=config['dataset'],
        name=config['algo'],
        mode=config['WANDB']['wandb_mode']
        # save_code=True,
        # job_type=config.function,
    )
    config = wandb.config
    print(config)
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)get_dataset
    from src.datasets.dataloader import get_dataset
    train_dl, test_dl = get_dataset(config, num_workers=4)
    from src.baselines.models import get_trainer
    trainer = get_trainer(config, train_dl, test_dl)

    # Create model directory and instantiate config.path
    get_experiment_dir(config)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        # Train the model
        import datetime

        print(datetime.datetime.now())
        trainer.fit(config.device)
        trainer.save_model_dict()
        # if config.algo == 'TimeVAE':
        #     save_obj(trainer.G.encoder.state_dict(), pt.join(
        #         config.exp_dir, 'encoder_state_dict.pt'))
        #     save_obj(trainer.G.decoder.state_dict(), pt.join(
        #         config.exp_dir, 'decoder_state_dict.pt'))
        # elif config.algo == 'TimeGAN':
        #     save_obj(trainer.recovery.state_dict(), pt.join(
        #         config.exp_dir, 'recovery_state_dict.pt'))
        #     save_obj(trainer.supervisor.state_dict(), pt.join(
        #         config.exp_dir, 'supervisor_state_dict.pt'))
        # else:
        #     save_obj(trainer.G.state_dict(), pt.join(
        #         config.exp_dir, 'generator_state_dict.pt'))

    elif config.pretrained:
        pass

    # Create the generative model, load the parameters and do evaluation
    from src.baselines.models import GENERATORS, VAES
    if config.algo == 'TimeGAN':
        generator = GENERATORS[config.generator](
            input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim,
            n_layers=config.G_num_layers, init_fixed=config.init_fixed)
        generator.load_state_dict(torch.load(pt.join(
            config.exp_dir, 'generator_state_dict.pt')), strict=True)

        supervisor = trainer.supervisor.to(
            device='cpu')
        supervisor.load_state_dict(torch.load(pt.join(
            config.exp_dir, 'supervisor_state_dict.pt')), strict=True)
        recovery = trainer.recovery.to(device='cpu')
        recovery.load_state_dict(torch.load(pt.join(
            config.exp_dir, 'recovery_state_dict.pt')), strict=True)
        recovery = nn.Sequential(supervisor, recovery)

        generator.eval()
        fake_test_dl = fake_loader(generator, num_samples=len(test_dl.dataset),
                                   n_lags=config.n_lags, batch_size=128, algo=config.algo, recovery=recovery
                                   )

        full_evaluation(generator, train_dl, test_dl,
                        config, recovery=recovery)
    elif config.algo == 'TimeVAE':
        vae = VAES[config.model](hidden_layer_sizes=config.hidden_layer_sizes,
                                 trend_poly=config.trend_poly,
                                 num_gen_seas=config.num_gen_seas,
                                 custom_seas=config.custom_seas,
                                 use_scaler=config.use_scaler,
                                 use_residual_conn=config.use_residual_conn,
                                 n_lags=config.n_lags,
                                 input_dim=config.input_dim,
                                 latent_dim=config.latent_dim,
                                 reconstruction_wt=config.reconstruction_wt)

        vae.encoder.load_state_dict(torch.load(pt.join(
            config.exp_dir, 'encoder_state_dict.pt')), strict=True)
        vae.decoder.load_state_dict(torch.load(pt.join(
            config.exp_dir, 'decoder_state_dict.pt')), strict=True)
        vae.eval()

        fake_test_dl = fake_loader(vae, num_samples=len(test_dl.dataset),
                                   n_lags=config.n_lags, batch_size=128, algo=config.algo)
        full_evaluation(vae, train_dl, test_dl, config)

    else:
        generator = GENERATORS[config.generator](
            input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim, n_layers=config.G_num_layers, init_fixed=config.init_fixed)
        generator.load_state_dict(torch.load(pt.join(
            config.exp_dir, 'generator_state_dict.pt')))

        fake_test_dl = fake_loader(generator, num_samples=len(test_dl.dataset),
                                   n_lags=config.n_lags, batch_size=test_dl.batch_size, algo=config.algo
                                   )
        full_evaluation(generator, train_dl, test_dl, config)

    # Plot the summary
    plot_summary(fake_test_dl, test_dl, config)
    # For non-stationary data, we plot the acf matrix
    if config.dataset == 'GBM' or config.dataset == 'ROUGH':
        compare_acf_matrix(test_dl, fake_test_dl, config)
    wandb.save(pt.join(config.exp_dir, '*png*'))
    wandb.save(pt.join(config.exp_dir, '*pt*'))
    wandb.save(pt.join(config.exp_dir, '*pdf*'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='RCGAN',
                        help='choose from TimeGAN,RCGAN,TimeVAE')
    parser.add_argument('--dataset', type=str, default='AR1',
                        help='choose from AR1, ROUGH, GBM,STOCK,Air_Quality')
    args = parser.parse_args()
    # if args.algo == 'TimeVAE':
    #     config_dir = 'configs/' + 'train_vae.yaml'
    #
    # else:
    #     config_dir = 'configs/' + 'train_gan.yaml'

    config_dir = 'configs/config.yaml'

    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    config.algo = args.algo

    config.dataset = args.dataset

    main(config)
