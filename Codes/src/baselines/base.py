import torch
import copy
from collections import defaultdict
import time
from src.utils import to_numpy
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from os import path as pt


class BaseTrainer:
    def __init__(self, batch_size, G, G_optimizer, test_metrics_train, test_metrics_test, n_gradient_steps, foo=lambda x: x):
        self.batch_size = batch_size

        self.G = G
        self.G_optimizer = G_optimizer
        self.n_gradient_steps = n_gradient_steps

        self.losses_history = defaultdict(list)

        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test
        self.foo = foo

        self.init_time = time.time()

        #self.best_G = copy.deepcopy(G.state_dict())
        self.best_G_loss = None
        #self.best_G = copy.deepcopy(self.G.state_dict())

    def evaluate(self, x_fake, x_real, step, config):
        self.losses_history['time'].append(time.time() - self.init_time)
        if config.algo == 'TimeGAN':
            x_fake = self.G(batch_size=1000,
                            n_lags=self.config.n_lags, condition=None, device=config.device)
            x_fake = self.recovery(self.supervisor(x_fake))

        if step % 200 == 0:
            with torch.no_grad():
                for test_metric in self.test_metrics_train:
                    test_metric(x_fake)
                    loss = to_numpy(test_metric.loss_componentwise)
                    if len(loss.shape) == 1:
                        loss = loss[..., None]
                    wandb.log(
                        {test_metric.name+'_train': loss, },
                        step=step,
                    )
                    self.losses_history[test_metric.name +
                                        '_train'].append(loss)
                for test_metric in self.test_metrics_test:
                    test_metric(x_fake)
                    loss = to_numpy(test_metric.loss_componentwise)
                    if len(loss.shape) == 1:
                        loss = loss[..., None]
                    wandb.log(
                        {test_metric.name+'_test': loss, },
                        step=step,
                    )
                    self.losses_history[test_metric.name +
                                        '_test'].append(loss)
        if step % 100 == 0:

            self.plot_sample(x_real, x_fake[:config.batch_size], self.config)
            wandb.log({'fake_samples': wandb.Image(
                pt.join(self.config.exp_dir, 'x_fake.png'))}, step)
            wandb.log({'real_samples': wandb.Image(
                pt.join(self.config.exp_dir, 'x_real.png'))}, step)
            torch.save(self.G.state_dict(),
                       pt.join(wandb.run.dir, 'generator_state_dict.pt'))

    @staticmethod
    def plot_sample(real_X, fake_X, config):
        sns.set()

        x_real_dim = real_X.shape[-1]
        for i in range(x_real_dim):
            plt.plot(to_numpy(fake_X[:250, :, i]).T, 'C%s' % i, alpha=0.1)
        plt.savefig(pt.join(config.exp_dir, 'x_fake.png'))
        plt.close()

        for i in range(x_real_dim):
            random_indices = torch.randint(0, real_X.shape[0], (250,))
            plt.plot(
                to_numpy(real_X[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
        plt.savefig(pt.join(config.exp_dir, 'x_real.png'))
        plt.close()
