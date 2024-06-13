import torch
from torch import nn
from autoencoder import AutoEncoder
from modules import Unet, UnetV2
from nnn_modules import DDPMScheduler, DDIMScheduler, TimeEmbed, ClassEmbed


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
        if unet_config['unet_version'] == 'v2':
            self.unet = UnetV2(**unet_config)
        else:
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
        z_pred = self(x, cls, t, cls_mask_ratio=0.15)
        return self.calculate_loss(z, z_pred)

    @torch.no_grad()
    def decode(self, x):
        return torch.clip(self.ae.decode(x * 4.5), -1, 1)  ### * 4.5 see data.TensorDataset

    @torch.no_grad()
    def encode(self, img):
        return self.ae.encode(img)

    @torch.no_grad()
    def condional_generation(self, cls, guidance_scale=1, batch_size=9):
        x = torch.randn([batch_size, self.latent_dim, self.latent_size, self.latent_size]).to(self.device)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            z_pred = self(x, cls, t)
            if guidance_scale > 1.0001:
                z_pred_unconditional = self(x, cls, t, cls_mask_ratio=1)
                z_pred = z_pred * guidance_scale + z_pred_unconditional * (1 - guidance_scale)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x)

    @torch.no_grad()
    def midway_generation(self, x0, cls, step_s=400, step_e=1000, batch_size=9):
        x0 = x0[:batch_size]
        cls = cls[:batch_size]
        step_s = int(step_s * self.sample_steps / 1000)
        step_e = min(self.sample_steps, int(step_e * self.sample_steps / 1000))
        z = torch.randn_like(x0)
        step_s_ = torch.ones(x0.shape[0]).long().to(self.device) * step_s
        x_ = self.sampler.diffuse(x0, self.sampler.step2t(step_s_), z)
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
            (torch.arange(batch_size)[:, None, None, None].to(self.device) / 2 / batch_size + 0.5)
        bias = torch.randn_like(x)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            z_pred = (x - self.sampler.alpha_bar_sqrt[t - 1] * x0) \
                     / (1 - self.sampler.alpha_bar[t - 1]).sqrt()
            z_pred = z_pred / (z_pred.var().sqrt() + 1e-5)
            z_pred += bias * 0.16 ** 0.5
            bias = 0.9 ** 0.5 * bias + 0.1 ** 0.5 * torch.randn_like(bias)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x)

    @torch.no_grad()
    def sim_training(self, x0, cls, batch_size=9):
        x0 = x0[:batch_size]
        cls = cls[:batch_size]
        z = torch.randn_like(x0)
        t = torch.randint(low=1, high=self.max_train_steps + 1, size=cls.shape).to(x0.device)
        x = self.sampler.diffuse(x0, t, z)
        z_pred = self(x, cls, t)
        x0_pred = self.sampler.rev_diffuse(x, t, z_pred)
        return self.decode(x), self.decode(x0_pred)

    def forward(self, x, cls, t, cls_mask_ratio=0):
        t = self.time_embed(t)
        cls = self.class_embed(cls)
        if len(t.shape) == 1:
            t = t.repeat((x.shape[0], 1))
        if len(cls.shape) == 1:
            cls = cls.repeat((x.shape[0], 1))
        if cls_mask_ratio > 0:
            cls[torch.rand(cls.shape[0]) < cls_mask_ratio] *= 0
        c = torch.cat((t, cls), dim=1)
        c = self.condition_embed(c)
        z_pred = self.unet(x, c)
        return z_pred


class LatentDiffusion2(LatentDiffusion):

    # loss = (x0 - x0_pred) ** 2

    def train_step(self, x0, cls):
        z = torch.randn_like(x0)
        t = torch.randint(low=1, high=self.max_train_steps + 1, size=cls.shape).to(x0.device)
        x = self.sampler.diffuse(x0, t, z)
        x0_pred = self(x, cls, t, cls_mask_ratio=0.15)
        return self.calculate_loss(x0, x0_pred)

    @torch.no_grad()
    def condional_generation(self, cls, guidance_scale=1, batch_size=9):
        x = torch.randn([batch_size, self.latent_dim, self.latent_size, self.latent_size]).to(self.device)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            x0_pred = self(x, cls, t)
            if guidance_scale > 1.0001:
                x0_pred_unconditional = self(x, cls, t, cls_mask_ratio=1)
                x0_pred = x0_pred * guidance_scale + x0_pred_unconditional * (1 - guidance_scale)
            z_pred = self.sampler.calc_z_pred(x, x0_pred, t)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x)

    @torch.no_grad()
    def validate_condional_generation(self, cls, guidance_scale=1, batch_size=20):
        x = torch.randn([batch_size, self.latent_dim, self.latent_size, self.latent_size]).to(self.device)
        x0 = torch.randn_like(x) + 0.3
        exp_noise = []
        pred_noise = []
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            t_ = torch.ones(x.shape[0]).long().to(self.device)*t
            z_scale = x - self.sampler.alpha_bar_sqrt[t_ - 1][:, None, None, None] * x0
            exp_noise.append((z_scale**2).mean().cpu().item())
            x_ = self.sampler.diffuse(x0, t_, torch.randn_like(x0))
            z_scale = x_ - self.sampler.alpha_bar_sqrt[t_ - 1][:, None, None, None] * x0
            pred_noise.append((z_scale**2).mean().cpu().item())
            z_pred = self.sampler.calc_z_pred(x, x0, t)
            x = self.sampler.step(x, z_pred, t, step)
        z_scale = x - x0
        exp_noise.append((z_scale ** 2).mean().cpu().item())
        z_scale = 0 * x0
        pred_noise.append((z_scale ** 2).mean().cpu().item())
        return exp_noise, pred_noise

    @torch.no_grad()
    def seq_condional_generation(self, cls, guidance_scale=1, n_steps=10, batch_size=3):
        seq_pred_x = []
        seq_x = []
        x = torch.randn([batch_size, self.latent_dim, self.latent_size, self.latent_size]).to(self.device)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            x0_pred = self(x, cls, t)
            if guidance_scale > 1.0001:
                x0_pred_unconditional = self(x, cls, t, cls_mask_ratio=1)
                x0_pred = x0_pred * guidance_scale + x0_pred_unconditional * (1 - guidance_scale)
            z_pred = self.sampler.calc_z_pred(x, x0_pred, t)
            x = self.sampler.step(x, z_pred, t, step)
            if (1 + step) % (self.sample_steps // n_steps) == 0:
                seq_pred_x.append(x0_pred)
                seq_x.append(x)
        seq_x = torch.cat(seq_x, dim=0)
        seq_pred_x = torch.cat(seq_pred_x, dim=0)
        return torch.cat([self.decode(seq_x), self.decode(seq_pred_x)], dim=0)

    @torch.no_grad()
    def halfway_condional_generation(self, cls, guidance_scale=1, batch_size=9, stop_t=200):
        x = torch.randn([batch_size, self.latent_dim, self.latent_size, self.latent_size]).to(self.device)
        for step in range(self.sample_steps):
            t = self.sampler.step2t(step)
            x0_pred = self(x, cls, t)
            if guidance_scale > 1.0001:
                x0_pred_unconditional = self(x, cls, t, cls_mask_ratio=1)
                x0_pred = x0_pred * guidance_scale + x0_pred_unconditional * (1 - guidance_scale)
            if t < stop_t:
                return self.decode(x0_pred)
            z_pred = self.sampler.calc_z_pred(x, x0_pred, t)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x)

    @torch.no_grad()
    def midway_generation(self, x0, cls, step_s=400, step_e=1000, batch_size=9):
        x0 = x0[:batch_size]
        cls = cls[:batch_size]
        step_s = int(step_s * self.sample_steps / 1000)
        step_e = min(self.sample_steps, int(step_e * self.sample_steps / 1000))
        z = torch.randn_like(x0)
        step_s_ = torch.ones(x0.shape[0]).long().to(self.device) * step_s
        x_ = self.sampler.diffuse(x0, self.sampler.step2t(step_s_), z)
        x = x_.detach()
        for step in range(step_s, step_e):
            t = self.sampler.step2t(step)
            x0_pred = self(x, cls, t)
            z_pred = self.sampler.calc_z_pred(x, x0_pred, t)
            x = self.sampler.step(x, z_pred, t, step)
        return self.decode(x_), self.decode(x)

    @torch.no_grad()
    def sim_training(self, x0, cls, batch_size=9):
        x0 = x0[:batch_size]
        cls = cls[:batch_size]
        z = torch.randn_like(x0)
        t = torch.randint(low=1, high=self.max_train_steps + 1, size=cls.shape).to(x0.device)
        x = self.sampler.diffuse(x0, t, z)
        x0_pred = self(x, cls, t)
        return self.decode(x), self.decode(x0_pred)
