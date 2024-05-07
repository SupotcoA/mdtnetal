import torch
from torch import nn
import torch.nn.functional as F
from nnn_modules import DDIMScheduler, DDPMScheduler, TimeEmbed, ClassEmbed


def cf2cl(tensor):
    return torch.permute(tensor, [0, 2, 3, 1])


def cl2cf(tensor):
    return torch.permute(tensor, [0, 3, 1, 2])


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=1)

    def forward(self, x, c=None):
        h = self.conv(x)
        return h


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        if kernel_size == 4:
            self.upsample = nn.Identity()
            self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           output_padding=0)

        elif kernel_size == 3:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding='same')

    def forward(self, x, c=None):
        h = self.upsample(x)
        h = self.conv(h)
        return h


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Sequential(nn.SiLU(),
                                nn.Linear(c_dim, 2 * n_channels, bias=True))

    def forward(self, x, c=None):
        beta = torch.mean(x, dim=(2, 3), keepdim=True)
        alpha = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False).sqrt()
        x = (x - beta) / (alpha + 1e-5)
        scale, bias = torch.chunk(self.fc(c), chunks=2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias)


class AdaptiveGroupNorm(nn.Module):

    def __init__(self, n_channels, c_dim, num_groups=32):
        super().__init__()
        self.num_groups = num_groups
        self.n_channels = n_channels
        self.fc = nn.Sequential(nn.SiLU(),
                                nn.Linear(c_dim, 2 * n_channels, bias=True))

    def forward(self, x, c=None):
        x = F.group_norm(x, self.num_groups, eps=1e-5)
        scale, bias = torch.chunk(self.fc(c), chunks=2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return x.mul(1 + scale).add(bias)


class ResBlock(nn.Module):

    def __init__(self, in_channels,
                 bottle_neck_channels=None,
                 out_channels=None,
                 res_bottle_neck_factor=2,
                 c_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        if bottle_neck_channels is not None:
            self.bottle_neck_channels = bottle_neck_channels
        else:
            self.bottle_neck_channels = int(max(self.out_channels,
                                                self.in_channels) \
                                            / res_bottle_neck_factor)
            self.bottle_neck_channels = max(32, self.bottle_neck_channels)

        self.norm1 = AdaptiveLayerNorm(n_channels=in_channels,
                                       c_dim=c_dim)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.bottle_neck_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=False)
        self.norm2 = AdaptiveLayerNorm(n_channels=self.bottle_neck_channels,
                                       c_dim=c_dim)
        self.conv2 = nn.Conv2d(in_channels=self.bottle_neck_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=True)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels=in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding='same')
        else:
            self.conv_shortcut = nn.Identity()
        # self.rescale = nn.Sequential(nn.ReLU(inplace=True),
        #                              nn.Linear(c_dim, out_channels, bias=True))

    def forward(self, x, c=None):
        h = x
        h = self.norm1(h, c)
        h = F.relu_(h)
        h = self.conv1(h)
        h = self.norm2(h, c)
        h = F.relu_(h)
        h = self.conv2(h)
        x = self.conv_shortcut(x)

        return x + h  # * self.rescale(c)


class AdaLNZeroResBlock(nn.Module):

    def __init__(self, in_channels,
                 bottle_neck_channels=None,
                 out_channels=None,
                 res_bottle_neck_factor=2,
                 c_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        if bottle_neck_channels is not None:
            self.bottle_neck_channels = bottle_neck_channels
        else:
            self.bottle_neck_channels = int(max(self.out_channels,
                                                self.in_channels) \
                                            / res_bottle_neck_factor)
            self.bottle_neck_channels = max(32, self.bottle_neck_channels)
        if self.bottle_neck_channels % 32 != 0:
            self.bottle_neck_channels = 32 * int(round(self.bottle_neck_channels/32))

        self.norm1 = torch.nn.GroupNorm(num_groups=32,
                                        num_channels=in_channels,
                                        eps=1e-6,
                                        affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.bottle_neck_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=False)
        self.norm2 = torch.nn.GroupNorm(num_groups=32,
                                        num_channels=self.bottle_neck_channels,
                                        eps=1e-6,
                                        affine=True)
        self.conv2 = nn.Conv2d(in_channels=self.bottle_neck_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=True)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels=in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding='same')
        else:
            self.conv_shortcut = nn.Identity()

        self.condition_proj = nn.Sequential(nn.ReLU(inplace=False),
                                            nn.Linear(c_dim, 5 * out_channels, bias=True),
                                            )

    def forward(self, x, c=None):
        alpha1, beta1, gamma1, beta2, gamma2 = torch.chunk(self.condition_proj(c), chunks=6, dim=1)
        h = x
        h = self.norm1(h).mul(1 + gamma1).add(beta1)
        h = F.relu_(h)
        h = self.conv1(h)
        h = self.norm2(h).mul(1 + gamma2).add(beta2)
        h = F.relu_(h)
        h = self.conv2(h)
        x = self.conv_shortcut(x)

        return x + h.mul(1 + alpha1[:, :, None, None])


class AttnBlock(nn.Module):
    def __init__(self, in_channels, embed_channels=None):
        super().__init__()
        self.in_channels = in_channels
        if embed_channels is not None:
            self.embed_channels = embed_channels
        else:
            self.embed_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels,
                                 eps=1e-6, affine=False)
        self.q = nn.Conv2d(in_channels,
                           self.embed_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv2d(in_channels,
                           self.embed_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv2d(in_channels,
                           self.embed_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv2d(self.embed_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def make_res_block(*args, **kwargs):
    # return ResBlock(*args,**kwargs)
    return AdaLNZeroResBlock(*args, **kwargs)


class Unet(nn.Module):

    # https://github.com/CompVis/taming-transformers/
    # blob/master/taming/modules/diffusionmodules/model.py#L195

    def __init__(self,
                 in_channels=3,
                 out_channels=None,
                 n_channels=64,
                 channels_mult=(1, 2, 2, 4),
                 num_res_blocks=2,
                 c_dim=None):
        super().__init__()

        ch = n_channels
        ch_mult = channels_mult
        out_ch = out_channels if out_channels is not None else in_channels

        self.ch = ch
        self.c_dim = c_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # norm before conv in?
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(make_res_block(in_channels=block_in,
                                            out_channels=block_out,
                                            c_dim=c_dim))
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = DownSample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res_block(in_channels=block_in,
                                          out_channels=block_in,
                                          c_dim=c_dim)
        self.mid.attn_1 = nn.Identity()  # AttnBlock(block_in)
        self.mid.block_2 = make_res_block(in_channels=block_in,
                                          out_channels=block_in,
                                          c_dim=c_dim)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(make_res_block(in_channels=block_in + skip_in,
                                            out_channels=block_out,
                                            c_dim=c_dim))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = AdaptiveLayerNorm(n_channels=block_in,
                                          c_dim=c_dim)

        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, c=None):

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], c)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, c)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, c)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop() / 1.4], dim=1), c)  # Imagen: shortcut/sqrt(2)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # assert len(hs)==0, f"len(hs) = {len(hs)}"
        # end
        # h = self.norm_out(h, c)
        h = F.relu_(h)
        h = self.conv_out(h)
        return h
