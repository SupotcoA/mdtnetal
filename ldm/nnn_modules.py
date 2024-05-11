import torch
from torch import nn
import numpy as np


class SchedulerBase(nn.Module):

    def __init__(self,
                 max_train_steps,
                 sample_steps=1000,
                 beta_schedule='cosine'):
        super(SchedulerBase, self).__init__()
        self.max_train_steps = max_train_steps
        self.sample_steps = sample_steps
        self.speedup_rate = int(round(max_train_steps / sample_steps))
        self.register_buffer('sample_t', torch.linspace(max_train_steps, 0, sample_steps + 1).long())
        if beta_schedule == 'cosine':
            s = 0.008
            x = torch.linspace(0, max_train_steps, max_train_steps + 1)
            alphas_cumprod = torch.cos(((x / max_train_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.register_buffer('beta', torch.clip(betas, 0.0001, 0.1))  ### 0.9999
        elif beta_schedule == 'linear':
            self.register_buffer('beta', torch.linspace(1e-4, 0.02, max_train_steps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_sqrt', self.alpha.sqrt())
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.register_buffer('alpha_bar_sqrt', self.alpha_bar.sqrt())
        self.register_buffer('sigma', self.beta.sqrt())
        assert self.beta.max() < 1 and self.beta.min() > 0
        assert self.alpha_bar[0] > 0.95 and self.alpha_bar[-1] < 0.05

    @torch.no_grad()
    def diffuse(self, x0, t: torch.Tensor, z):
        x = self.alpha_bar_sqrt[t - 1][:, None, None, None] * x0 \
            + (1 - self.alpha_bar[t - 1]).sqrt()[:, None, None, None] * z
        return x

    @torch.no_grad()
    def rev_diffuse(self, x, t: torch.Tensor, z):
        x0 = (x - (1 - self.alpha_bar[t - 1]).sqrt()[:, None, None, None] * z) \
             / self.alpha_bar_sqrt[t - 1][:, None, None, None]
        return x0


class DDPMScheduler(SchedulerBase):

    def __init__(self,
                 max_train_steps,
                 sample_steps=1000,
                 beta_schedule='cosine'):
        super(DDPMScheduler, self).__init__(max_train_steps=max_train_steps,
                                            sample_steps=sample_steps,
                                            beta_schedule=beta_schedule)

    @torch.no_grad()
    def step2t(self, step):
        return self.max_train_steps - step

    @torch.no_grad()
    def step(self, x, z_pred, t, step=None):  # t = 1~1000
        z = torch.randn_like(z_pred) if t > 1 else 0
        x = (x - (1 - self.alpha[t - 1]) / (1 - self.alpha_bar[t - 1]).sqrt() * z_pred) / \
            self.alpha_sqrt[t - 1] + self.sigma[t - 1] * z
        return x


class DDIMScheduler(SchedulerBase):

    def __init__(self,
                 max_train_steps,
                 sample_steps=50,
                 beta_schedule='cosine'):
        super(DDIMScheduler, self).__init__(max_train_steps=max_train_steps,
                                            sample_steps=sample_steps,
                                            beta_schedule=beta_schedule)

    @torch.no_grad()
    def step2t(self, step):
        return self.sample_t[step]

    @torch.no_grad()
    def mean_pred(self, x, z_pred, t, t_prev):
        if t_prev == 0:
            alpha_bar_sqrt_prev = 1
        else:
            alpha_bar_sqrt_prev = self.alpha_bar_sqrt[t_prev - 1]
        return alpha_bar_sqrt_prev * (x - (1 - self.alpha_bar[t - 1]).sqrt() * z_pred)\
               / self.alpha_bar_sqrt[t - 1]

    @torch.no_grad()
    def std_pred(self, t_prev):
        if t_prev == 0:
            return 1
        else:
            return (1 - self.alpha_bar[t_prev - 1]).sqrt()

    @torch.no_grad()
    def step(self, x, z_pred, t, step):  # step = 0~self.sample_steps-1
        assert t == self.sample_t[step], f"{t} {step} {self.sample_t[step]}"
        t_prev = self.sample_t[step + 1]
        x = self.mean_pred(x, z_pred, t, t_prev) + z_pred * self.std_pred(t_prev)
        return x


class TimeEmbed(nn.Module):

    def __init__(self,
                 embed_dim,
                 max_train_steps=1000,
                 learnable=False):
        super(TimeEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.max_train_steps = max_train_steps
        embed = self.get_sinusoidal(embed_dim, max_train_steps)
        self.embed = nn.Parameter(embed, requires_grad=learnable)

    @staticmethod
    def get_sinusoidal(embed_dim, max_train_steps):
        timesteps = torch.arange(1, 1+max_train_steps)
        half_dim = embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embed_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, t):
        if isinstance(t, int):
            return self.embed[t - 1]
        elif isinstance(t, torch.Tensor):
            return self.embed[t.long() - 1]
        else:
            raise TypeError(f"t is type {type(t)}")


class ClassEmbed(nn.Module):

    def __init__(self,
                 embed_dim,
                 n_classes=1000):
        super(ClassEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.max_train_steps = n_classes
        embed = torch.zeros([n_classes, embed_dim], dtype=torch.float32)
        self.embed = nn.Parameter(embed, requires_grad=True)

    def forward(self, cls):  # e.g. cls = 0~2
        if isinstance(cls, int):
            return self.embed[cls]
        elif isinstance(cls, torch.Tensor):
            return self.embed[cls.long()]
        else:
            raise TypeError(f"cls is type {type(cls)}")