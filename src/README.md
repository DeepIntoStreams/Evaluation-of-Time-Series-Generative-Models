Code base for evaluation of time series generation 
========================================

This repository is the code base for evaluation of time series generation 

## Environment Setup
The code has been tested successfully using Python 3.8 and pytorch 1.11.0. A typical process for installing the package dependencies involves creating a new Python virtual environment.

To install the required packages, run the following:
```console
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall
pip install cupy-cuda102
pip install git+https://github.com/tgcsaba/ksig.git --no-deps
pip install -r requirements.txt
```
Note that this pipeline relies on **wandb** for logging and tracking experiemnts, user needs to create a personal wandb account and specify the personal wandb api key in [configs/train_gan.yaml](configs/train_gan.yaml). 


## Description 

 We aim to create an pipeline with easy configuration interface for controlling on various GAN baseline models, datasets and evaluation metrics. The pipeline is still under development and may change based on the objective of the paper, it currently support 3 generative models on three different datasets. 
 One should able to run experiments with
 
 ```console
 python run.py --algo ALGO_NAME --dataset DATA
 ```
 where `ALGO_NAME` is a choice of `TimeGAN`, `RCGAN`.`TimeVAE`. ` DATA` is a choice `AR1`, `ROUGH`,`GBM`.
 The [run.py](run.py) will retrieve configuration from [configs/train_gan.yaml](configs/train_gan.yaml) and complete the model training on the specified dataset and evaluation. One can modify [configs/train_gan.yaml](configs/train_gan.yaml) to further specify model parameters.
 

After the model training and evaluation, the related model and plots will be store in the [numerical_results](numerical_results) folder, these files will also uploaded the wandb online folder for each individual run. 
 