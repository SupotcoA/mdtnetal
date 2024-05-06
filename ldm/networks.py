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
                                         'cosine')
        else:
            self.sampler = DDIMScheduler(self.max_train_steps,
                                         self.sample_steps,
                                         'cosine')
        embed_dim = unet_config['c_dim']
        self.time_embed = TimeEmbed(embed_dim=embed_dim,
                                    max_train_steps=diffusion_config['max_train_steps'])
        class_embed_dim = min(data_config['n_classes'], embed_dim)
        self.class_embed = ClassEmbed(embed_dim=class_embed_dim,
                                      n_classes=data_config['n_classes'])
        self.condition_embed = nn.Sequential(nn.Linear(class_embed_dim + embed_dim,
                                                       embed_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(embed_dim, embed_dim))

    @staticmethod
    def calculate_loss(z, z_pred):
        return (z - z_pred).pow(2).mean()

    def train_step(self, x0, cls):
        z = torch.randn(x0.shape, dtype=x0.dtype).to(x0.device)
        t = torch.randint(low=1, high=self.max_train_steps + 1, size=cls.shape).to(x0.device)
        x = self.sampler.diffuse(x0, t, z)
        z_pred = self(x, cls, t)
        return self.calculate_loss(z, z_pred)

    @torch.no_grad()
    def decode(self, x):
        return self.ae.decode(x) / 0.1

    @torch.no_grad()
    def encode(self, img):
        return self.ae.encode(img) * 0.1

    @torch.no_grad()
    def condional_generation(self, cls, batch_size=9):
        if isinstance(cls, int):
            cls = torch.ones(batch_size).long().to(self.device) * cls
        x = torch.randn([batch_size, self.latent_dim, *self.latent_size]).to(self.device)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            z_pred = self(x, cls, t)
            x = self.sampler.step(x, z_pred, t, step)
        return torch.clip(self.decode(x), -1, 1)

    def forward(self, x, cls, t):
        t = self.time_embed(t)
        cls = self.class_embed(cls)
        c = torch.cat((t, cls), dim=1)
        z_pred = self.unet(x, c)
        return z_pred
