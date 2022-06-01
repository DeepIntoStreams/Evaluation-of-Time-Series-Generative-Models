Code base for evaluation of time series generation 
========================================

This repository is the code base for evaluation of time series generation 

## Environment Setup
The code has been tested successfully using Python 3.8; thus we suggest using this version or a later version of Python. A typical process for installing the package dependencies involves creating a new Python virtual environment.

To install the required packages, run the following:
```console
pip install .
```
Note that this pipeline relis on **wandb** for logging experiemnts, user needs to create a personal wandb account. 

## Description 

 We aim to create an pipeline with easy configuration interface for controlling on various GAN baseline models, datasets and evaluation metrics. The pipeline is still under development and may change based on the objective of the paper, also it would great to have progressive adjustments during our work.
Here I provid a top down description of the pipeline for the latest version. On the top level, one should able to run experiments with
 
 ```console
 python run.py
 ```
 The [run.py](run.py) will retrieve configuration from [config/train_gan.py](config/train_gan.yaml) and complete the model training on the specified dataset and evaluation. The config yaml file essentially specify the model, dataset and related hyperparameters for the experiment. (I should add a functionality to specify the evaluation metrics later, however this may require more thinking since the evaluation have dependency on the model type and dataset.)
NOTE: user need to replace the wandb api key and wandb init message inside the run.py to run the experiment in their own wandb account.

There are three subdirectories inside the [src/](src/) folder corresponding to models, datasets and evaluations:
1. [src/baselines/](src/baselines/) contains baseline GAN models in separate py files and a master file [src/baselines/models.py](src/baselines/models.py) for retrieving models based on the config.

2. [src/datasets/](src/datasets/) contains the data generation in separate py files and a master file [src/datasets/dataloader.py](src/datasets/dataloader.py) to create the pytroch dataloder based on the config. The intermediate processed data in tensor format will be store in [data/](data/) to avoid repeatedly downloading and data processing. 

3. [src/evaluations](src/evaluations/) contains test metrics and evaluations such as discriminative score and various test metrics on financial time series from Sig-wgan paper. (##TODO should add a master file allow user to retrieve the test metrics based on the config)

I hope this description this clear enough for the users to add individual model, dataset and evaluation method in a similar way for the future code development.  

After the model training and evaluation, the related model and plots will be store in the [numerical_experiments](numerical_experiemtns) folder, these files will also uploaded the wandb online folder for each run. The output files of repeated the experiments on the same dataset and model will be replaced in [numerical_experiments](numerical_experiemtns) folder, but the output will be uploaded separately on wandb for each run. 

