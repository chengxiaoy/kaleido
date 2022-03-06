import numpy as np

from model import BaseVAE, EncoderBottleneck, DecoderBottleneck
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from abc import abstractmethod
from typing import Any, List
from math import sqrt
from style_gan_util import *
from soft_intro_vae import DownResLayer


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = EqualConv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        return x


class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False, fused_scale=True):
        super(EncodeBlock, self).__init__()
        self.conv_1 = EqualConv2d(inputs, inputs, 3, 1, 1)
        # self.conv_1 = EqualConv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = Blur(inputs)
        self.last = last
        self.fused_scale = fused_scale
        if last:
            self.dense = EqualLinear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = EqualConv2d(inputs, outputs, 3, 2, 1)
            else:
                self.conv_2 = EqualConv2d(inputs, outputs, 3, 1, 1)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)
        self.style_1 = EqualLinear(2 * inputs, latent_size)
        if last:
            self.style_2 = EqualLinear(outputs, latent_size)
        else:
            self.style_2 = EqualLinear(2 * outputs, latent_size)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)

        x = self.instance_norm_1(x)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))

            x = F.leaky_relu(x, 0.2)
            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(x.view(x.shape[0], x.shape[1]))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2

            x = F.leaky_relu(x, 0.2)

            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
            style_2 = torch.cat((m, std), dim=1)

            x = self.instance_norm_2(x)

            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))

        return x, w1, w2


class Encoder(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Encoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, False, fused_scale=fused_scale)

            resolution //= 2

            # print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


class ALAE(BaseVAE):
    def __init__(self, code_dim=512, n_mlp=8, resolution=128):
        super(ALAE, self).__init__()

        t = [PixelNorm()]
        for i in range(n_mlp):
            t.append(EqualLinear(code_dim, code_dim))
            t.append(nn.LeakyReLU(0.2))
        self.trans = nn.Sequential(*t)

        self.generator = Generator(code_dim=code_dim)

        self.encoder = Encoder(12, 512, layer_count=int(np.log2(128)) - 1, latent_size=code_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


if __name__ == '__main__':
    resolution = 128
    lay_count = int(np.log2(128)) - 1
    encoder = Encoder(16, 512, layer_count=lay_count, latent_size=512)
    x = torch.randn(32, 3, resolution, resolution)
    w = encoder(x, lay_count - 1, 0)
    print(w.shape)
    print(torch.flatten(w, start_dim=1))
