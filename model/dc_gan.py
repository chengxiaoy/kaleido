from typing import Any

from model.types_ import Tensor
from torch import nn
import torch


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, in_channels, latent_dim, feature_maps):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # state size. (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # state size. (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # state size. (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # state size. (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (in_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels, latent_dim, feature_maps):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class DCGAN(nn.Module):

    def __init__(self, **params):
        # 将 VanillaVAE 类型的self 转换成为BaseVAE 并执行基类的init方法
        super(DCGAN, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]
        self.feature_maps = params["feature_maps"]
        self.batch_size = params["batch_size"]

        self.discriminator = Discriminator(self.in_channel, self.latent_dim, self.feature_maps)
        self.generator = Generator(self.in_channel, self.latent_dim, self.feature_maps)

        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        z = torch.randn(inputs.shape[0], self.latent_dim, 1, 1).to(inputs.device)
        generate_x = self.generator(z)
        fake_label = self.discriminator(generate_x).view(-1)
        real_label = self.discriminator(inputs).view(-1)
        labels = torch.zeros(self.batch_size).to(inputs.device)
        return [fake_label, real_label, labels, generate_x]

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        fake_label = inputs[0]
        real_label = inputs[1]
        labels = inputs[2]
        real = 1.
        fake = 0.

        discriminator_loss = nn.BCELoss()(fake_label, labels.clone().fill_(fake)) \
                             + nn.BCELoss()(real_label, labels.clone().fill_(real))

        generator_loss = nn.BCELoss()(fake_label, labels.clone().fill_(real))

        return {"loss": discriminator_loss + generator_loss, "d_loss": discriminator_loss, "g_loss": generator_loss}

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[3]
