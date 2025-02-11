import torch
from src.models.base import BaseTrainer
from tqdm import tqdm
import wandb
from os import path as pt
from src.utils import save_obj


class TimeVAETrainer(BaseTrainer):
    def __init__(self, G, train_dl, config,
                 **kwargs):
        super(TimeVAETrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.train_dl = train_dl
        self.conditional = self.config.conditional

    def fit(self, device):
        self.G.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)

    def step(self, device, step):
        # generate x_fake

        if self.conditional:
            data = next(iter(self.train_dl))
            x = data[0].to(device)
            condition = data[1].to(device)
            # condition = one_hot(
            #     data[1], self.config.num_classes).unsqueeze(1).repeat(1, data[0].shape[1], 1).to(device)
            x_real_batch = torch.cat(
                [x, condition], dim=2).permute([0, 2, 1])
        else:
            condition = None
            x_real_batch = next(iter(self.train_dl))[
                0].permute([0, 2, 1]).to(device)

        G_loss = self.G_trainstep(device, x_real_batch, step)
        wandb.log({'G_loss': G_loss}, step)

    def G_trainstep(self, device, x_real, step):

        latent_z, mean, log_var = self.G.encoder(x_real)
        self.toggle_grad(self.G, True)
        self.G.encoder.train()
        self.G.decoder.train()
        self.G_optimizer.zero_grad()
        x_fake = self.G.decoder(latent_z)

        reconstruction_loss, latent_loss = self.compute_loss(
            x_real, x_fake, mean, log_var)
        G_loss = self.G.reconstruction_wt * reconstruction_loss + latent_loss
        G_loss.backward()
        self.losses_history['reconstruction_loss'].append(reconstruction_loss)
        self.losses_history['latent_loss'].append(latent_loss)
        self.G_optimizer.step()

        # self.evaluate(x_fake,x_real, step,self.config) # Need on for

        return G_loss.item()

    def compute_loss(self, x_real, x_fake, mean, log_var):
        latent_loss = -0.5 * \
            (1. + log_var - torch.pow(mean, 2) - torch.exp(log_var)).mean()
        reconstruction_loss = torch.norm(x_real - x_fake, dim=0).mean()
        return reconstruction_loss, latent_loss

    def save_model_dict(self):
        save_obj(self.G.encoder.state_dict(), pt.join(
            self.config.exp_dir, 'encoder_state_dict.pt'))
        save_obj(self.G.decoder.state_dict(), pt.join(
            self.config.exp_dir, 'decoder_state_dict.pt'))
