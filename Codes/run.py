"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import ml_collections
import copy
import wandb
import yaml
import os

from os import path as pt
from typing import Optional
import argparse
from src.evaluations.evaluations import compute_discriminative_score, fake_loader, compute_classfication_score, plot_summary
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils import get_experiment_dir, save_obj



def main():
    config_dir = 'configs/' + 'train_gan.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    # print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    # initialize weight and bias
    # Place here your API key.
    # os.environ["WANDB_API_KEY"] = "a0a43a4b820d0a581e3579b07d15bd9881f4b559"
    tags = [
        config.algo,
        config.dataset,
    ]

    wandb.init(
        project='Generative_model_evaluation',
        config=copy.deepcopy(dict(config)),
        entity="jiajie0502",
        tags=tags,
        group=config.dataset,
        name=config.algo
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

    # Define transforms and create dataloaders

    # WandB – wan
    # from src.datasets.dataloader import db.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log="all", log_freq=200) # -> There was a wandb bug that made runs in Sweeps crash

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
        save_obj(trainer.G.state_dict(), pt.join(
            config.exp_dir, 'generator_state_dict.pt'))

    elif config.pretrained:
        pass

        # Select test function
        #_test_CIFAR10(model, test_loader, config)
    from src.baselines.models import GENERATORS
    generator = GENERATORS[config.generator](
        input_dim=config.G_input_dim, hidden_dim=config.G_hidden_dim, output_dim=config.input_dim, n_layers=config.G_num_layers)
    generator.load_state_dict(torch.load(pt.join(
        config.exp_dir, 'generator_state_dict.pt')))
    fake_train_dl = fake_loader(generator, num_samples=len(train_dl.dataset),
                                n_lags=config.n_lags, batch_size=train_dl.batch_size, config=config
                                )
    fake_test_dl = fake_loader(generator, num_samples=len(test_dl.dataset),
                               n_lags=config.n_lags, batch_size=test_dl.batch_size, config=config
                               )
    if config.dataset in ['ROUGH', 'STOCK']:
        plot_summary(fake_test_dl, test_dl, config)
    else:
        pass
    wandb.save(pt.join(config.exp_dir, '*png*'))
    wandb.save(pt.join(config.exp_dir, '*pt*'))
    wandb.save(pt.join(config.exp_dir, '*pdf*'))
    discriminative_score = compute_discriminative_score(
        train_dl, test_dl, fake_train_dl, fake_test_dl, config, epochs=50)
    wandb.run.summary['discriminative_score'] = discriminative_score
    if config.dataset == 'sMnist':
        TFTR_acc, TRTF_acc = compute_classfication_score(train_dl, fake_train_dl, config,
                                                         hidden_size=64, num_layers=3, epochs=50)
        wandb.run.summary['TFTR_acc'] = TFTR_acc
        wandb.run.summary['TRTF_acc'] = TRTF_acc
    else:
        pass


if __name__ == '__main__':
    main()
