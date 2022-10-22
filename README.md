# Evaluation of Time Series Generative Models

The main objective of the project is to summarize the evaluation metrics used in unconditional generative models for synthetic data generation, list advantages and disadvantages of each evaluation metric based on experiments on different datasets and models. We also provide a code pipeline for evaluation of test metrics. 

## Test metrics
We include the following test metric in our code pipeline:
- Sig-W1 metric: a generic metric for distribution induced by time series.  
- Metrics on marginal distribution: to measure the fitting of generative models in terms of the fitting of the marginal distribution.  
- Metrics on dependency: to measure the fitting of generative models in terms of correlation and autocorrelation.  
- Discriminative score: to train a classifier to distinguish whether the sample is from the true distribution or synthetic distribution. The smaller the discriminative score, the better generator.  
- Predictive score: train a sequence-to-sequence model to predict the latter part of a time series given the first part, using generated data, then evaluate on the true data. Smaller losses, meaning ability to predict, are better.  

## Models
We implement some well known models including:
- Time-GAN
- Recurrent Conditional GAN (RCGAN)
- Time-VAE 

## Datasets
We provide the following datasets for model testing:
- Autoregressive process
- Geometric Brownian motion
- Rough volatility model (rough Bergomi)
- Google stock data
- BeiJing air quality data


## Project Outcomes
- A survey paper;
- A Python code base;
- A database (library) for relevant papers (allow the update from the community).

