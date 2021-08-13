from model import BaseVAE, DecoderBottleneck
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from typing import List, Any


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AGE(BaseVAE):

    def __init__(self, **params):
        super(AGE, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]
        self.batch_size = params["batch_size"]
        self.gamma = params["gamma"]
        self.mu = params["mu"]

        in_channels = self.in_channel
        fm_dim = 64

        # down_sample 32X
        self.encoder = nn.Sequential(
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

            nn.Conv2d(fm_dim * 8, self.latent_dim, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(self.latent_dim)
        )

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

    def encode(self, input: Tensor) -> List[Tensor]:
        z = self.encoder(input)
        z = torch.flatten(z, start_dim=1)
        return z

    def decode(self, input: Tensor) -> Any:
        result = self.decoder_input(input)
        result = result.view(-1, 512, self.image_size // 32, self.image_size // 32)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        labels = torch.zeros(self.batch_size).to(x.device)
        z = self.encode(x)
        z = F.normalize(z)
        x__ = self.decode(z)

        z__ = torch.randn(z.shape).to(x.device)
        z__ = F.normalize(z__)
        x_ = self.decode(z__)
        z_ = self.encode(x_)
        z_ = F.normalize(z_)

        return [x, x_, x__, z, z_, z__]

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        x = inputs[0]
        x_ = inputs[1]
        x__ = inputs[2]
        z = inputs[3]
        z_ = inputs[4]
        z__ = inputs[5]

        # d_loss = KLN01Loss(True)(z) + KLN01Loss(False)(z_) + match(x, x_, "L1")
        x_cons = self.mu * match(x, x__, "L1")
        z_cons = self.gamma * match(z_, z__, "L2")
        d_loss = KLN01Loss(True)(z) + KLN01Loss(False)(z_) + x_cons
        g_loss = KLN01Loss(True)(z_) + z_cons

        return {"loss": d_loss + g_loss, "gan_g_loss": g_loss, "gan_d_loss": d_loss, "z_cons": z_cons, "x_cons": x_cons}

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[2]


def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class KLN01Loss(torch.nn.Module):

    def __init__(self, minimize, direction='qp'):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), 'wtf?'

        samples = samples.view(samples.size(0), -1)

        self.samples_var = var(samples)
        self.samples_mean = samples.mean(0)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            # mu_1 = 0; sigma_1 = 1

            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # mu_2 = 0; sigma_2 = 1

            t1 = (samples_var.pow(2) + samples_mean.pow(2)) / 2
            t2 = -samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1

        return KL


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
