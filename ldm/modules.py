import torch
from torch import nn
import torch.nn.functional as F


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


class AdaptiveUnknownNorm(nn.Module):

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


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, n_channels, c_dim):
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Sequential(nn.SiLU(),
                                nn.Linear(c_dim, 2 * n_channels, bias=True))

    def forward(self, x, c=None):
        beta = torch.mean(x, dim=1, keepdim=True)
        alpha = torch.var(x, dim=1, keepdim=True, unbiased=False).sqrt()
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


class ResBlockV2(nn.Module):

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
                self.bottle_neck_channels = 32 * (self.bottle_neck_channels // 32 + 1)

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.bottle_neck_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=False)
        self.norm2 = AdaptiveGroupNorm(n_channels=self.bottle_neck_channels,
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
        h = self.norm1(h)
        h = F.relu_(h)
        h = self.conv1(h)
        h = self.norm2(h, c)
        h = F.relu_(h)
        h = self.conv2(h)
        x = self.conv_shortcut(x)

        return x + h  # * self.rescale(c)


class SemiConvNeXtBlock(nn.Module):

    def __init__(self, in_channels,
                 bottle_neck_channels=None,
                 out_channels=None,
                 res_bottle_neck_factor=2,  # ConvNeXt: 4
                 c_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        if bottle_neck_channels is not None:
            self.bottle_neck_channels = bottle_neck_channels
        else:
            self.bottle_neck_channels = int(max(self.out_channels,
                                                self.in_channels) \
                                            * res_bottle_neck_factor)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=5,  # ConvNeXt: 7
                               stride=1,
                               padding='same',
                               bias=False,
                               groups=in_channels)  # ConvNeXt: depth-wise
        self.norm = AdaptiveLayerNorm(n_channels=in_channels,
                                      c_dim=c_dim)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.bottle_neck_channels,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=self.bottle_neck_channels,
                               out_channels=self.out_channels,
                               kernel_size=1,
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

    def forward(self, x, c=None):
        h = x
        h = self.conv1(h)
        h = self.norm(h, c)
        h = self.conv2(h)
        h = F.gelu(h)
        h = self.conv3(h)
        x = self.conv_shortcut(x)

        return x + h


class AdaLNZeroResBlock(nn.Module):

    def __init__(self, in_channels,
                 bottle_neck_channels=None,
                 out_channels=None,
                 res_bottle_neck_factor=2,
                 c_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.norm1 = torch.nn.GroupNorm(num_groups=32,
                                        num_channels=in_channels,
                                        eps=1e-6,
                                        affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=False)
        self.norm2 = torch.nn.GroupNorm(num_groups=32,
                                        num_channels=in_channels,
                                        eps=1e-6,
                                        affine=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
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
                                            nn.Linear(c_dim, 5 * in_channels, bias=True),
                                            )

    ### channel bugs here!!!
    def forward(self, x, c=None):
        alpha1, beta1, gamma1, beta2, gamma2 = torch.chunk(self.condition_proj(c), chunks=5, dim=1)
        h = x
        h = self.norm1(h).mul(1 + gamma1[:, :, None, None]).add(beta1[:, :, None, None])
        h = F.relu_(h)
        h = self.conv1(h)
        h = self.norm2(h).mul(1 + gamma2[:, :, None, None]).add(beta2[:, :, None, None])
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
                                 eps=1e-6, affine=True)
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


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / torch.sqrt(torch.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, embed_channels=None, head_channels=-1, num_heads=1):
        super().__init__()
        self.channels = in_channels
        if head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    in_channels % head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {head_channels}"
            self.num_heads = in_channels // head_channels
        self.norm = nn.GroupNorm(32, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        h = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(h))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


def make_res_block(*args, **kwargs):
    return ResBlock(*args, **kwargs)
    # return AdaLNZeroResBlock(*args, **kwargs)
    # return SemiConvNeXtBlock(*args, **kwargs)


def make_res_block_v2(*args, **kwargs):
    return ResBlockV2(*args, **kwargs)


def make_attn_block(in_channels, embed_channels=None, head_channels=None):
    if head_channels is None:
        return AttnBlock(in_channels, embed_channels)
    else:
        return MultiHeadAttnBlock(in_channels, embed_channels=None, head_channels=None)


class Unet(nn.Module):

    # https://github.com/CompVis/taming-transformers/
    # blob/master/taming/modules/diffusionmodules/model.py#L195

    def __init__(self,
                 in_channels=3,
                 out_channels=None,
                 n_channels=96,
                 channels_mult=(1, 2, 4),
                 num_res_blocks=2,
                 res_bottle_neck_factor=2,
                 c_dim=None,
                 **ignoredkeys):
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
        curr_res = 32
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
                                            res_bottle_neck_factor=res_bottle_neck_factor,
                                            c_dim=c_dim))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = DownSample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res_block(in_channels=block_in,
                                          out_channels=block_in,
                                          res_bottle_neck_factor=res_bottle_neck_factor,
                                          c_dim=c_dim)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = make_res_block(in_channels=block_in,
                                          out_channels=block_in,
                                          res_bottle_neck_factor=res_bottle_neck_factor,
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
                                            res_bottle_neck_factor=res_bottle_neck_factor,
                                            c_dim=c_dim))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in)
                curr_res = curr_res * 2
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
        h = self.norm_out(h, c)
        h = F.relu_(h)
        h = self.conv_out(h)
        return h


class UnetV2(nn.Module):

    # https://github.com/CompVis/taming-transformers/
    # blob/master/taming/modules/diffusionmodules/model.py#L195

    def __init__(self,
                 in_channels=3,
                 out_channels=None,
                 n_channels=96,
                 channels_mult=(1, 2, 4),
                 num_res_blocks=2,
                 res_bottle_neck_factor=2,
                 c_dim=None,
                 attn_resolutions=(8, 16),
                 **ignoredkeys):
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
        curr_res = 32
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(make_res_block_v2(in_channels=block_in,
                                               out_channels=block_out,
                                               res_bottle_neck_factor=res_bottle_neck_factor,
                                               c_dim=c_dim))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_channels=32))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = DownSample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res_block_v2(in_channels=block_in,
                                             out_channels=block_in,
                                             res_bottle_neck_factor=res_bottle_neck_factor,
                                             c_dim=c_dim)
        self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_channels=32)
        self.mid.block_2 = make_res_block_v2(in_channels=block_in,
                                             out_channels=block_in,
                                             res_bottle_neck_factor=res_bottle_neck_factor,
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
                block.append(make_res_block_v2(in_channels=block_in + skip_in,
                                               out_channels=block_out,
                                               res_bottle_neck_factor=res_bottle_neck_factor,
                                               c_dim=c_dim))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_channels=32))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(32, block_in)

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
        h = self.norm_out(h, c)
        h = F.relu_(h)
        h = self.conv_out(h)
        return h
