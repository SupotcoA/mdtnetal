import torch
from torch import nn
from autoencoder import AutoEncoder
from modules import Unet, DDPMScheduler, DDIMScheduler, TimeEmbed, ClassEmbed


class LatentDiffusion(nn.Module):

    def __init__(self,
                 data_config,
                 unet_config,
                 ae_config,
                 diffusion_config):
        super(LatentDiffusion, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_size = ae_config['latent_size']
        self.latent_dim = ae_config['latent_dim']
        self.max_train_steps = diffusion_config['max_train_steps']
        self.sample_steps = diffusion_config['sample_steps']
        self.unet = Unet(**unet_config)
        self.ae = AutoEncoder(**ae_config)
        if self.max_train_steps == self.sample_steps:
            self.sampler = DDPMScheduler(self.max_train_steps,
                                         self.sample_steps,
                                         diffusion_config['strategy'])
        else:
            self.sampler = DDIMScheduler(self.max_train_steps,
                                         self.sample_steps,
                                         diffusion_config['strategy'])
        embed_dim = unet_config['c_dim']
        self.time_embed = TimeEmbed(embed_dim=embed_dim,
                                    max_train_steps=diffusion_config['max_train_steps'])
        class_embed_dim = min(data_config['n_classes'], embed_dim)
        self.class_embed = ClassEmbed(embed_dim=class_embed_dim,
                                      n_classes=data_config['n_classes'])
        self.condition_embed = nn.Sequential(nn.Linear(class_embed_dim + embed_dim,
                                                       embed_dim),
                                             nn.SiLU(),
                                             nn.Linear(embed_dim, embed_dim))

    @staticmethod
    def calculate_loss(z, z_pred):
        return (z - z_pred).pow(2).mean()

    def train_step(self, x0, cls):
        z = torch.randn_like(x0)
        t = torch.randint(low=1, high=self.max_train_steps + 1, size=cls.shape).to(x0.device)
        x = self.sampler.diffuse(x0, t, z)
        z_pred = self(x, cls, t)
        return self.calculate_loss(z, z_pred)

    @torch.no_grad()
    def decode(self, x):
        return torch.clip(self.ae.decode(x / 0.1 / 3), -1, 1)  ### /3 see data.TensorDataset

    @torch.no_grad()
    def encode(self, img):
        return self.ae.encode(img) * 0.1

    @torch.no_grad()
    def condional_generation(self, cls, batch_size=9):
        x = torch.randn([batch_size, self.latent_dim, self.latent_size, self.latent_size]).to(self.device)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            z_pred = self(x, cls, t)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x)

    @torch.no_grad()
    def midway_generation(self, x0, cls, step_s=400, step_e=1000, batch_size=9):
        step_s = int(step_s * self.sample_steps / 1000)
        step_e = min(self.sample_steps, int(step_e * self.sample_steps / 1000))
        z = torch.randn_like(x0)
        step_s_ = torch.ones(x0.shape[0]).long().to(self.device)*step_s
        x_ = self.sampler.diffuse(x0, self.sampler.step2t(step_s_),z)
        x = x_.detach()
        for step in range(step_s, step_e):
            t = self.sampler.step2t(step)
            z_pred = self(x, cls, t)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x_), self.decode(x)

    @torch.no_grad()
    def validate_generation(self, x0, batch_size=9):
        x0 = x0[:batch_size]
        x = torch.randn_like(x0) * \
            (torch.arange(batch_size)[:, None, None, None].to(self.device)/2/batch_size+0.5)
        bias = torch.randn_like(x)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            z_pred = (x - self.sampler.alpha_bar_sqrt[t - 1] * x0) \
                      / (1 - self.sampler.alpha_bar[t - 1]).sqrt()
            z_pred = z_pred / (z_pred.var().sqrt()+1e-5)
            z_pred += bias * 0.16**0.5
            bias = 0.9**0.5*bias + 0.1**0.5*torch.randn_like(bias)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x)

    @torch.no_grad()
    def sim_training(self, x0, cls, batch_size=9):
        z = torch.randn_like(x0)
        t = torch.randint(low=1, high=self.max_train_steps + 1, size=cls.shape).to(x0.device)
        x = self.sampler.diffuse(x0, t, z)
        z_pred = self(x, cls, t)
        x0_pred = self.sampler.rev_diffuse(x, t, z_pred)
        return self.decode(x), self.decode(x0_pred)

    def forward(self, x, cls, t):
        t = self.time_embed(t)
        cls = self.class_embed(cls)
        if len(t.shape) == 1:
            t = t.repeat((x.shape[0], 1))
        if len(cls.shape) == 1:
            cls = cls.repeat((x.shape[0], 1))
        c = torch.cat((t, cls), dim=1)
        c = self.condition_embed(c)
        z_pred = self.unet(x, c)
        return z_pred
