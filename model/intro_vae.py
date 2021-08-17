from model import BaseVAE
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from typing import List, Any


class ResBlock(nn.Module):

    def __init__(self, kernel_size_list: List, in_channel, out_channel):
        super(ResBlock, self).__init__()

        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )

        modules = []
        for kernel_size in kernel_size_list:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channel)
            ))
            in_channel = out_channel
        self.conv = nn.Sequential(*modules)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        identity = self.down_sample(identity)
        out += identity
        out = self.relu(self.bn(out))
        return out


class DownResLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size_list: List):
        super(DownResLayer, self).__init__()
        self.res_layer = ResBlock(kernel_size_list, in_channel, out_channel)
        self.down_sample = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.res_layer(x)
        out = self.down_sample(out)
        return out


class UpResLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size_list: List):
        super(UpResLayer, self).__init__()
        self.res_layer = ResBlock(kernel_size_list, in_channel, out_channel)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        out = self.up_sample(x)
        out = self.res_layer(out)
        return out


class INTRO_VAE(BaseVAE):

    def __init__(self, **params):
        super(INTRO_VAE, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]
        self.batch_size = params["batch_size"]
        self.alpha = params["alpha"]
        self.m = params["m"]
        self.beta = params["beta"]

        # down_sample 32X
        self.encoder = nn.Sequential(
            DownResLayer(3, 16, [5]),
            DownResLayer(16, 32, [1, 3, 3]),
            DownResLayer(32, 64, [1, 3, 3]),
            DownResLayer(64, 128, [1, 3, 3]),
            DownResLayer(128, 256, [1, 3, 3]),
        )

        self.fc_mu = nn.Linear(256 * (self.image_size // 32) ** 2, self.latent_dim)
        self.fc_var = nn.Linear(256 * (self.image_size // 32) ** 2, self.latent_dim)

        # up_sample 32X
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * (self.image_size // 32) ** 2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            UpResLayer(256, 256, [3, 3]),
            UpResLayer(256, 128, [3, 3]),
            UpResLayer(128, 64, [1, 3, 3]),
            UpResLayer(64, 32, [1, 3, 3]),
            UpResLayer(32, 16, [1, 3, 3]),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(16, out_channels=self.in_channel,
                      kernel_size=5, padding=2),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        z = self.encoder(input)
        z = torch.flatten(z, start_dim=1)
        mu = self.fc_mu(z)
        logvar = self.fc_var(z)
        return [mu, logvar]

    def decode(self, input: Tensor) -> Any:
        result = self.decoder_input(input)
        result = result.view(-1, 256, self.image_size // 32, self.image_size // 32)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        labels = torch.zeros(self.batch_size).to(x.device)
        mu, logvar = self.encode(x)
        prior_loss = self.kl_loss(mu, logvar)
        z = self.reparameter_trick(mu, logvar)
        recons_x = self.decode(z)
        reconstruction_loss = self.reconstruction_loss(recons_x, x, True)

        recons_mu, recons_logvar = self.encode(recons_x)
        recons_reg_loss = self.kl_loss(recons_mu, recons_logvar)

        sample_z = torch.randn(z.shape).to(x.device)
        sample_x = self.decode(sample_z)
        sample_mu, sample_logvar = self.encode(recons_x)
        sample_reg_loss = self.kl_loss(sample_mu, sample_logvar)

        return [x, recons_x, sample_x, reconstruction_loss, prior_loss, recons_reg_loss, sample_reg_loss]

    def reparameter_trick(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(logvar)
        return mu + eps * std

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        x = inputs[0]
        recons_x = inputs[1]
        sample_x = inputs[2]
        recons_loss = inputs[3]
        reg_loss = inputs[4]
        recons_reg_loss = inputs[5]
        sample_reg_loss = inputs[6]

        encoder_loss = reg_loss + self.alpha * (
                F.relu(self.m - sample_reg_loss) + F.relu(self.m - recons_reg_loss)) * 0.5 + self.beta * recons_loss

        decoder_loss = self.alpha * (sample_reg_loss + recons_reg_loss) * 0.5 + self.beta * recons_loss

        return {"loss": encoder_loss + decoder_loss, "decoder_loss": decoder_loss, "encoder_loss": encoder_loss,
                "recons_loss": recons_loss}

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]

    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        return v_kl.mean()

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error


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
