# Evaluation of Time Series Generative Models

The main objective of the project is to summarize the evaluation metrics used in unconditional generative models for synthetic data generation, list advantages and disadvantages of each evaluation metric based on experiments on different datasets and models. We also provide a code pipeline for evaluation of test metrics. 

## Test metrics
We include the following test metric in our code pipeline:
- Sig-W1 metric [1]: a generic metric for distribution induced by time series.  
- Metrics on marginal distribution [1]: to measure the fitting of generative models in terms of the fitting of the marginal distribution.  
- Metrics on dependency [1]: to measure the fitting of generative models in terms of correlation and autocorrelation.  
- Discriminative score [2]: to train a classifier to distinguish whether the sample is from the true distribution or synthetic distribution. The smaller the discriminative score, the better generator.  
- Predictive score [2]: train a sequence-to-sequence model to predict the latter part of a time series given the first part, using generated data, then evaluate on the true data. Smaller losses, meaning ability to predict, are better.  

## Models
We implement some popular models for time series generation including:
- Time-GAN [2]
- Recurrent Conditional GAN (RCGAN) [3]
- Time-VAE [4]

## Datasets
We provide the following datasets for model testing:
- Autoregressive process
- Geometric Brownian motion
- Rough volatility model (rough Bergomi)
- Google stock data [2]
- Beijing air quality data [5]

## Structure
The repository is structures as follows: 
- Codes: contains the code pipeline for evaluation metrics of time series generation, implementation of different well known generative models and their corresponding training procedure.
- Reference Papers: contains the collection of papers related to time series generation.

## Reference
[1] Ni, H., Szpruch, L., Wiese, M., Liao, S. and Xiao, B., 2021. Sig-Wasserstein GANs for Time Series Generation.  
[2] Yoon, J., Jarrett, D. and Van der Schaar, M., 2019. Time-series generative adversarial networks. Advances in neural information processing systems, 32.  
[3] Esteban, C., Hyland, S.L. and Rätsch, G., 2017. Real-valued (medical) time series generation with recurrent conditional gans. arXiv preprint arXiv:1706.02633.  
[4] Desai A., Freeman C., Wang, Z.H., Beaver I., 2021 TimeVAE: A Variational Auto-Encoder For Multivariate Time Series Generation. arXiv preprint arXiv:2111.08095.  
[5] Zhang S., Guo B., Dong A., He J., Xu Z., and Chen S.X., 2017, Cautionary tales on air-quality improvement in beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences.


