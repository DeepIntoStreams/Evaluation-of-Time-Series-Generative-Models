from setuptools import setup, find_packages

setup(
    name="Evalation-of-time-series-generation",
    version="0.0.1",
    author="DeepIntoStreams",
    description="Codebase for evaluation of time series generation ",
    url="https://github.com/DeepIntoStreams/Evaluation-of-Time-Series-Generative-Models/tree/main/Codes",
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['matplotlib == 3.5.2',
                      'ml_collections == 0.1.1',
                      'numpy',
                      'PyYAML == 6.0',
                      'sklearn',
                      'torch == 1.11.0',
                      'signatory == 1.2.6.1.9.0',
                      'torchvision == 0.12.0',
                      'seaborn',
                      'tqdm==4.64.0',
                      'wandb',
                      'wfdb==3.4.1',
                      'fbm==0.3.0',
                      'evaluate',
                      'lib==4.0.0',
                      'pandas==1.3.5',
                      'requests==2.27.1']
)
