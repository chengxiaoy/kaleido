from model.base import *
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Linear(in_channels, out_channels),
                                      nn.ReLU(True),
                                      nn.Linear(out_channels, out_channels))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels, fm_dim, latent_dim):
        super(Discriminator, self).__init__()
        self.x_layer = nn.Sequential(
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
            # nn.BatchNorm2d(fm_dim * 4),

        )

        self.z_layer = nn.Sequential(
            ResidualLayer(latent_dim, latent_dim),
            ResidualLayer(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, fm_dim * 4),
        )

        self.final_layer = nn.Sequential(
            nn.Linear(fm_dim * 4, 1),
            nn.LeakyReLU()
        )

    def forward(self, x, z):
        x_code = self.x_layer(x).squeeze(dim=-1).squeeze(dim=-1)
        z_code = self.z_layer(z)
        x_code = x_code + z_code
        output = self.final_layer(x_code).squeeze(dim=-1)
        return output


class VEE_GAN(BaseVAE):
    """
    adversarially learned inference
    """

    def __init__(self, **params):
        super(VEE_GAN, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]
        self.batch_size = params["batch_size"]
        self.beta = params["beta"]

        # down_sample 32X
        self.encoder = nn.Sequential(
            EncoderBottleneck(self.in_channel, 32, 3, 2, 1),
            EncoderBottleneck(32, 64, 3, 2, 1),
            EncoderBottleneck(64, 128, 3, 2, 1),
            EncoderBottleneck(128, 256, 3, 2, 1),
            EncoderBottleneck(256, 512, 3, 2, 1),
        )

        self.fc_mu = nn.Linear(512 * (self.image_size // 32) ** 2, self.latent_dim)
        self.fc_var = nn.Linear(512 * (self.image_size // 32) ** 2, self.latent_dim)

        # up_sample 32X
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

        # discriminator
        self.discriminator = Discriminator(3, 64, 128)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        # cause fc_var is a linear layer, the output value range is [-inf,inf], so it can not represent the variance
        # but can indicate the log value of variance, i guess
        # logvar = self.fc_var(result)
        # return [mu, logvar]
        return mu

    def decode(self, input: Tensor) -> Any:
        result = self.decoder_input(input)
        result = result.view(-1, 512, self.image_size // 32, self.image_size // 32)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        labels = torch.zeros(self.batch_size).to(x.device)
        z_ = self.encode(x)

        z = torch.randn(z_.shape).to(device=x.device)
        x_ = self.decode(z)
        z__ = self.encode(x_)

        real_label = self.discriminator(x, z_)
        fake_label = self.discriminator(x_, z)

        return [x, x_, z, z__, real_label, fake_label, labels]

    def reparameter_trick(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(logvar)
        return mu + std * eps

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        x = inputs[0]
        x_ = inputs[1]
        z = inputs[2]
        z__ = inputs[3]
        real_label = inputs[4]
        fake_label = inputs[5]
        labels = inputs[6]
        # prior regulation
        real = 1.
        fake = 0.
        # print("fake=======================")
        # print(fake_label)
        # print("real=======================")
        # print(real_label)
        dis_loss = nn.BCEWithLogitsLoss()(fake_label, labels.clone().fill_(fake)) \
                   + nn.BCEWithLogitsLoss()(real_label, labels.clone().fill_(real))

        decoder_loss = nn.BCEWithLogitsLoss()(fake_label, labels.clone().fill_(real)) + self.beta * match(z, z__, "L2")

        encoder_loss = self.beta * match(z, z__, "L2")

        return {"loss": dis_loss + decoder_loss + encoder_loss, "dis_loss": dis_loss, "decoder_loss": decoder_loss,
                "encoder_loss": encoder_loss}

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]

    if __name__ == "__main__":
        dis = Discriminator(3, 64, 128)
        x = torch.randn(32, 3, 128, 128)
        z = torch.randn(32, 128)
        output = dis(x, z)
        print(output)


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = F.normalize(x)
        y_n = F.normalize(y)

        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'
