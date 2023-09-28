from os import path as pt
from src.evaluations.evaluations import compute_discriminative_score, fake_loader, compute_classfication_score, \
    full_evaluation
from src.evaluations.plot import plot_summary, compare_acf_matrix
import torch
from src.utils import get_experiment_dir, save_obj
from torch import nn
import ml_collections
import copy
import wandb
import yaml
import os
from src.utils import set_seed


def get_test_default_config(test_config='test/test_config.yaml',model_type=None):
    with open(test_config) as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    config.algo = 'TimeGAN'
    config.dataset = 'GBM'
    os.environ['WANDB_START_METHOD']='thread'
    config.steps = 5

    if model_type is not None:
        if model_type == 'TimeGAN' or model_type == 'TimeVAE':
            for k,v in config[model_type].items():
                config[k] = v
        else:
            raise Exception("Model type not supported")
    return config

def update_config(config, **kwargs):
    for k,v in kwargs.items():
        config[k] = v
    return config

def test_init(config=None):
    if config is None:
        config = get_test_default_config()

    if config.WANDB.wandb_api is not None:
        os.environ["WANDB_API_KEY"] = config.WANDB.wandb_api
        tags = [
        config.algo,
        config.dataset,
        'test'
    ]
        wandb.init(
            project='Generative_model_evaluation',
            config=copy.deepcopy(dict(config)),
            entity="deepintostreams",
            tags=tags,
            group=config.dataset,
            name=config.algo,
            mode=config.WANDB.wandb_mode
        )
        # config = wandb.config
        
        if (config.device == "cuda" and torch.cuda.is_available()):
            config.update({"device": "cuda:0"}, allow_val_change=True)
            if torch.cuda.is_available():
                if config.seed is not None:
                    set_seed(config.seed)
                    torch.cuda.manual_seed_all(config.seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
        else:
            config.update({"device": "cpu"}, allow_val_change=True)
            if config.seed is not None:
                set_seed(config.seed)

        return config
    # else:
    #     raise Exception("Wandb API key is not provided")
    


