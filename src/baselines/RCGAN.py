import torch
from torch import autograd

from src.baselines.base import BaseTrainer
from tqdm import tqdm
from torch.nn.functional import one_hot
import wandb
import torch.nn.functional as F


class RCGANTrainer(BaseTrainer):
    def __init__(self, D, G, train_dl, config,
                 **kwargs):
        super(RCGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(
            D.parameters(), lr=config.lr_D, betas=(0, 0.9))  # Using TTUR

        self.train_dl = train_dl
        self.conditional = self.config.conditional
        self.reg_param = 0
        self.losses_history

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)

    def step(self, device, step):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            if self.conditional:
                data = next(iter(self.train_dl))
                x = data[0].to(device)
                condition = one_hot(
                    data[1], self.config.num_classes).unsqueeze(1).repeat(1, data[0].shape[1], 1).to(device)
                x_real_batch = torch.cat(
                    [x, condition], dim=2)
            else:
                condition = None
                x_real_batch = next(iter(self.train_dl))[0].to(device)
            with torch.no_grad():
                x_fake = self.G(batch_size=self.batch_size,
                                n_lags=self.config.n_lags, condition=condition, device=device)

            D_loss_real, D_loss_fake = self.D_trainstep(
                x_fake, x_real_batch)
            if i == 0:
                self.losses_history['D_loss_fake'].append(D_loss_fake)
                self.losses_history['D_loss_real'].append(D_loss_real)
                self.losses_history['D_loss'].append(D_loss_fake + D_loss_real)
                wandb.log({'D_loss': D_loss_fake + D_loss_real}, step)
        G_loss = self.G_trainstep(x_real_batch, device, step)
        wandb.log({'G_loss': G_loss}, step)

    def G_trainstep(self, x_real, device, step):
        if self.conditional:
            condition = one_hot(torch.randint(
                0, self.config.num_classes, (self.batch_size,)),
                self.config.num_classes).float().unsqueeze(1).repeat(1, self.config.n_lags, 1).to(device)

        else:
            condition = None
        x_fake = self.G(batch_size=self.batch_size,
                        n_lags=self.config.n_lags, condition=condition, device=device)
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        self.D.train()
        G_loss = self.compute_loss(d_fake, 1.)
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.G.parameters(), 10)
        self.losses_history['G_loss'].append(G_loss)
        self.G_optimizer.step()
        self.evaluate(x_fake, x_real, step, self.config)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1.)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0.)

        # Compute regularizer on fake / real
        dloss = dloss_fake + dloss_real

        dloss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.D.parameters(), 10)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return dloss_real.item(), dloss_fake.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(
            batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
