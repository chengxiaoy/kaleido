from typing import Any, List

from model import EncoderBottleneck, DecoderBottleneck, BaseVAE
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, in_channels, fm_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels, fm_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(fm_dim, fm_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(fm_dim * 2, fm_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(fm_dim * 4, fm_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(fm_dim * 8, fm_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fm_dim * 8, fm_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fm_dim * 4),

        )

        self.final_layer = nn.Sequential(
            nn.Linear(fm_dim * 4, 1),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )

    def forward(self, input):
        dis_code = self.main(input).squeeze(dim=-1).squeeze(dim=-1)
        output = self.final_layer(dis_code).squeeze(dim=-1)
        return dis_code, output


class VAEGAN(BaseVAE):

    def __init__(self, **params):
        # 将 VanillaVAE 类型的self 转换成为BaseVAE 并执行基类的init方法
        super(VAEGAN, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]
        self.batch_size = params["batch_size"]

        # 将图片down sample 32 倍
        self.encoder = nn.Sequential(
            EncoderBottleneck(self.in_channel, 32, 3, 2, 1),
            EncoderBottleneck(32, 64, 3, 2, 1),
            EncoderBottleneck(64, 128, 3, 2, 1),
            EncoderBottleneck(128, 256, 3, 2, 1),
            EncoderBottleneck(256, 512, 3, 2, 1),
        )

        self.fc_mu = nn.Linear(512 * (self.image_size // 32) ** 2, self.latent_dim)
        self.fc_var = nn.Linear(512 * (self.image_size // 32) ** 2, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, 512 * (self.image_size // 32) ** 2)

        self.decoder = nn.Sequential(
            DecoderBottleneck(512, 256, 3, 2, 1, 1),
            DecoderBottleneck(256, 128, 3, 2, 1, 1),
            DecoderBottleneck(128, 64, 3, 2, 1, 1),
            DecoderBottleneck(64, 32, 3, 2, 1, 1),
            DecoderBottleneck(32, 32, 3, 2, 1, 1),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(32, out_channels=self.in_channel,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.dis = Discriminator(3, 64)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        # cause fc_var is a linear layer, the output value range is [-inf,inf], so it can not represent the variance
        # but can indicate the log value of variance, i guess
        logvar = self.fc_var(result)
        return [mu, logvar]

    def decode(self, input: Tensor) -> Any:
        result = self.decoder_input(input)
        result = result.view(-1, 512, self.image_size // 32, self.image_size // 32)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        labels = torch.zeros(self.batch_size).to(inputs.device)
        mu, logvar = self.encode(inputs)
        latent_code = self.reparameter_trick(mu, logvar)
        outputs = self.decode(latent_code)

        sample_z = torch.randn(self.batch_size, self.latent_dim).to(outputs.device)
        sample_x = self.decode(sample_z)
        _, sample_label = self.dis(sample_x)

        real_code, real_label = self.dis(inputs)
        fake_code, fake_label = self.dis(outputs)

        return [inputs, outputs, mu, logvar, labels, fake_label, real_label, sample_label, fake_code, real_code]

    def reparameter_trick(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(logvar)
        return mu + eps * std

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        input = inputs[0]
        output = inputs[1]
        mu = inputs[2]
        logvar = inputs[3]
        labels = inputs[4]
        fake_label = inputs[5]
        real_label = inputs[6]
        sample_label = inputs[7]
        fake_code = inputs[8]
        real_code = inputs[9]
        # prior regulation
        prior_loss = torch.mean(torch.sum(-0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)), dim=1), dim=0)
        # reconstruction_loss build on discriminator feature

        reconstruction_loss = F.mse_loss(real_code, fake_code, size_average=False) / input.size(0)
        # gan loss
        # Establish convention for real and fake labels during training
        real = 1.
        fake = 0.
        gan_loss = nn.BCELoss()(sample_label, labels.clone().fill_(fake)) \
                   + nn.BCELoss()(fake_label, labels.clone().fill_(fake)) \
                   + nn.BCELoss()(real_label, labels.clone().fill_(real))
        loss = prior_loss + reconstruction_loss + gan_loss

        return {"loss": loss, "reconstruction_loss": reconstruction_loss, "prior_loss": prior_loss,
                "gan_loss": gan_loss}

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]


if __name__ == '__main__':
    input = torch.randn(32, 3, 64, 64)
    discriminator = Discriminator(3, 64)
    dis_code, output = discriminator(input)
    print(dis_code.size())
    print(output.size())
