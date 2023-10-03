# Evaluation of Time Series Generative Models

The main objective of the project is to summarize the evaluation metrics used in unconditional generative models for synthetic data generation, list the advantages and disadvantages of each evaluation metric based on experiments on different datasets and models. 


## Test metrics
We include the following test metric in our code pipeline:
- Sig-W1 metric [1]: a generic metric for distribution induced by time series.  
- Metrics on marginal distribution [1]: to measure the fitting of generative models in terms of the marginal distribution.  
- Metrics on dependency [1]: to measure the fitting of generative models in terms of correlation and autocorrelation.  
- Discriminative score [2]: to train a classifier to distinguish whether the sample is from the true distribution or synthetic distribution. The smaller the discriminative score, the better generator.  
- Predictive score [2]: train a sequence-to-sequence model to predict the latter part of a time series given the first part, using generated data, then evaluate on the true data. Smaller losses, meaning ability to predict, are better.  
- Distance-based metrics [6]: this metrics concerns the distance between real and generated samples. Can be used to assess the diversity and fidelity of generated data, furthermore, it can determine a potential model collapse.
- Permutation tests [7]: perform a permutation test using signature-based MMD to obtain the power of the test.
- t-SNE plot: a statistical method for visualizing high-dimensional data embedded into 2-dimensional data based on Stochastic Neighbor Embedding.


We also provide a code pipeline that provides the implementation of well-known generative models, their corresponding training procedure, and model assessment using the evaluation test metrics described before. Furthermore, we provide several datasets as examples to demonstrate how to utilize the pipeline. See the figure below for the workflow of our pipeline:

![Code pipeline](Pipeline.png)

We list the datasets and models we have implemented in this repository:

## Datasets
We provide the following datasets for model testing:
- Autoregressive process;
- Geometric Brownian motion;
- Rough volatility model (rough Bergomi);
- Google stock data [2];
- Beijing air quality data [5].

## Models
We implement some popular models for time series generation including:
- Time-GAN [2];
- Recurrent Conditional GAN (RCGAN) [3];
- Time-VAE [4].

For detailed instructions on repository structure and usage, refer to [Codes/README.md](Codes/README.md).
## Reference
[1] Ni, H., Szpruch, L., Wiese, M., Liao, S. and Xiao, B., 2021. Sig-Wasserstein GANs for Time Series Generation.  
[2] Yoon, J., Jarrett, D. and Van der Schaar, M., 2019. Time-series generative adversarial networks. Advances in neural information processing systems, 32.  
[3] Esteban, C., Hyland, S.L. and Rätsch, G., 2017. Real-valued (medical) time series generation with recurrent conditional gans. arXiv preprint arXiv:1706.02633.  
[4] Desai A., Freeman C., Wang, Z.H., Beaver I., 2021. TimeVAE: A Variational Auto-Encoder For Multivariate Time Series Generation. arXiv preprint arXiv:2111.08095.  
[5] Zhang S., Guo B., Dong A., He J., Xu Z., and Chen S.X., 2017. Cautionary tales on air-quality improvement in Beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences.  
[6] Anonymous authors, 2022. SOK: The Great GAN Bake Off, An Extensive Systematic Evaluation Of Generative Adversarial Network Architectures For Time Series.  
[7] Chevyrev I. and Oberhauser H., 2022. Signature moments to characterize laws of stochastic processes.  

## Citation
Please use the following to cite our work if you find it useful.

```
@Misc{EvalTimeseriesGen2023,
author =   {Hang Lou, Jiajie Tao, Xin Dong, Baoren Xiao, Lei Jiang, and Hao Ni},
title =    {Evaluation of Time Series Generative Models},
howpublished = {https://github.com/DeepIntoStreams/Evaluation-of-Time-Series-Generative-Models.git},
year = {2023}
}
```
