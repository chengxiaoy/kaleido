from typing import List, Any

import torch
import torch.nn.functional as F
from torch import nn

from model import BaseVAE
from model.types_ import Tensor


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
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2)
            ))
            in_channel = out_channel
        self.conv = nn.Sequential(*modules)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.2)

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


class SOFT_INTRO_VAE(BaseVAE):

    def __init__(self, **params):
        super(SOFT_INTRO_VAE, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]
        self.batch_size = params["batch_size"]
        self.beta_rec = params["beta_rec"]
        self.beta_kl = params["beta_kl"]
        self.beta_neg = params["beta_neg"]
        self.gamma_r = params["gamma_r"]
        self.scale = 1 / (3 * self.image_size ** 2)

        # down_sample 64X
        self.encoder = nn.Sequential(
            DownResLayer(3, 16, [5]),
            DownResLayer(16, 32, [1, 3, 3]),
            DownResLayer(32, 64, [1, 3, 3]),
            DownResLayer(64, 128, [1, 3, 3]),
            DownResLayer(128, 256, [1, 3, 3]),
            DownResLayer(256, 256, [1, 3, 3]),
        )

        self.fc_mu = nn.Linear(256 * (self.image_size // 64) ** 2, self.latent_dim)
        self.fc_var = nn.Linear(256 * (self.image_size // 64) ** 2, self.latent_dim)

        # up_sample 32X
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * (self.image_size // 64) ** 2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            UpResLayer(256, 256, [3, 3]),
            UpResLayer(256, 128, [3, 3]),
            UpResLayer(128, 64, [1, 3, 3]),
            UpResLayer(64, 32, [1, 3, 3]),
            UpResLayer(32, 16, [1, 3, 3]),
            UpResLayer(16, 16, [1, 3, 3]),
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
        result = result.view(-1, 256, self.image_size // 64, self.image_size // 64)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def elbo(self, x):
        mu, logvar, z, recons_x = self.forward(x)
        kl_loss = self.kl_loss(mu, logvar)
        reconstruction_loss = self.reconstruction_loss(recons_x, x, True)
        return kl_loss, reconstruction_loss, recons_x, z

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameter_trick(mu, logvar)
        recons_x = self.decode(z)
        return mu, logvar, z, recons_x

    def reparameter_trick(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(logvar)
        return mu + eps * std

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[3]

    def kl_loss(self, mu, logvar, prior_mu=0, mean=True):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        if mean:
            return v_kl.mean()
        else:
            return v_kl.sum()

    def reconstruction_loss(self, prediction, target, size_average=True):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error

    def calc_reconstruction_loss(self, x, recon_x, loss_type='mse', reduction='sum'):
        """

        :param x: original inputs
        :param recon_x:  reconstruction of the VAE's input
        :param loss_type: "mse", "l1", "bce"
        :param reduction: "sum", "mean", "none"
        :return: recon_loss
        """
        if reduction not in ['sum', 'mean', 'none']:
            raise NotImplementedError
        recon_x = recon_x.view(recon_x.size(0), -1)
        x = x.view(x.size(0), -1)
        if loss_type == 'mse':
            recon_error = F.mse_loss(recon_x, x, reduction='none')
            recon_error = recon_error.sum(1)
            if reduction == 'sum':
                recon_error = recon_error.sum()
            elif reduction == 'mean':
                recon_error = recon_error.mean()
        elif loss_type == 'l1':
            recon_error = F.l1_loss(recon_x, x, reduction=reduction)
        elif loss_type == 'bce':
            recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
        else:
            raise NotImplementedError
        return recon_error

    def calc_kl(self, logvar, mu, mu_o=10, is_outlier=False, reduce='sum'):
        """
        Calculate kl-divergence
        :param logvar: log-variance from the encoder
        :param mu: mean from the encoder
        :param mu_o: negative mean for outliers (hyper-parameter)
        :param is_outlier: if True, calculates with mu_neg
        :param reduce: type of reduce: 'sum', 'none'
        :return: kld
        """
        if is_outlier:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp() + 2 * mu * mu_o - mu_o.pow(2)).sum(1)
        else:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        if reduce == 'sum':
            kl = torch.sum(kl)
        elif reduce == 'mean':
            kl = torch.mean(kl)
        return kl


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
