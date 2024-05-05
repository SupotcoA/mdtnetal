from diffusers.models import AutoencoderKL
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self,
                 model_type="stabilityai/sd-vae-ft-ema",
                 # params = 34 + 49 M
                 # f = 8; embed_dim = 4
                 **ignoredkeys):

        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type).eval().requires_grad_(False)

    def forward(self, x):
        return self.model(x).sample

    def encode(self, x, mode=True):
        dist = self.model.encode(x).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    def decode(self, x):
        return self.model.decode(x).sample