import numpy as np
import pytorch_lightning
import torch
from torch import nn
from torch.autograd import grad

from model.style_gan_util import Generator, F, PixelNorm, EqualConv2d, EqualLinear, ConvBlock
from model.soft_intro_vae import DownResLayer
import random


class Trans(nn.Module):
    def __init__(self, n_mlp, code_dim):
        super(Trans, self).__init__()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, x):
        return self.style(x)


class Encoder(nn.Module):

    def __init__(self, code_dim, from_rgb_activate=False, fused=True):
        super(Encoder, self).__init__()
        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, code_dim)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, code_dim):
        super(Discriminator, self).__init__()
        self.linear = EqualLinear(code_dim, 1)

    def forward(self, x):
        return self.linear(x)


class ALAE(pytorch_lightning.LightningModule):

    def __init__(self, code_dim, n_mlp=8, **param):
        super(ALAE, self).__init__()

        self.t = Trans(n_mlp, code_dim)

        self.generator = Generator(code_dim)

        self.encoder = Encoder(code_dim=code_dim)

        self.discriminator = Discriminator(code_dim=code_dim)

    def trans(self, z):
        return self.t(z)

    def generate(self, w, batch, noise=None,
                 step=0, mean_style=None, style_weight=0,
                 mixing_range=(-1, -1), alpha=-1):

        styles = w

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=styles[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def encode(self, x, step, alpha):
        return self.encoder(x, step, alpha)

    def discriminate(self, w):
        return self.discriminator(w)


if __name__ == '__main__':
    resolution = 128
    batch_size = 32
    code_dim = 512
    step = int(np.log2(resolution)) - 2
    z = torch.randn(batch_size, code_dim)
    param = {"n_mlp": 8}
    model = ALAE(code_dim=code_dim, **param)
    w = model.trans(z)
    x = model.generate([w], batch=batch_size, step=step)
    w_ = model.encode(x, step=step,alpha=1)
    y = model.discriminate(w_)
    print(z.size())
    print(w.size())
    print(w_.size())
    print(x.size())
    print(y.size())
