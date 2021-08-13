from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from model import flow, BaseVAE, EncoderBottleneck, DecoderBottleneck
import numpy as np

from typing import Any, List


class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        # bernoulli log prob is equivalent to negative binary cross entropy
        return -self.bce_with_logits(logits, target)


class NF_VAE(BaseVAE):
    """
    normalizing flows   objective

    elbo = log(p(x))+KL(q(z|x)||p(z|x))
         = \sum{p(x|z)} + KL(q(z|x)||p(z))
         = \sum{p(x,z)} + H(q(z|x))
    """

    def __init__(self, **params):
        super(NF_VAE, self).__init__()

        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.latent_dim = params["latent_dim"]

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
        self.fc_h = nn.Linear(512 * (self.image_size // 32) ** 2, self.latent_dim)

        modules = []
        for _ in range(2):
            modules.append(flow.InverseAutoregressiveFlow(num_input=self.latent_dim,
                                                          num_hidden=self.latent_dim * 2,
                                                          num_context=self.latent_dim))
            modules.append(flow.Reverse(self.latent_dim))
        self.q_z_flow = flow.FlowSequential(*modules)

        self.log_p_z = NormalLogProb()
        self.log_q_z_0 = NormalLogProb()
        self.register_buffer('p_z_loc', torch.zeros(self.latent_dim))
        self.register_buffer('p_z_scale', torch.ones(self.latent_dim))

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
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = result.unsqueeze(dim=1)
        mu = self.fc_mu(result)
        # cause fc_var is a linear layer, the output value range is [-inf,inf], so it can not represent the variance
        # but can indicate the log value of variance, i guess
        logvar = self.fc_var(result)
        h = self.fc_h(result)
        scale = torch.exp(0.5 * logvar)
        z_0 = self.reparameter_trick(mu, logvar)

        log_q_z_0 = self.log_q_z_0(mu, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)

        return [z_T.squeeze(dim=1), log_q_z.squeeze(dim=1)]

    def reparameter_trick(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(logvar)
        return mu + eps * std

    def decode(self, input: Tensor) -> Any:
        result = self.decoder_input(input)
        result = result.view(-1, 512, self.image_size // 32, self.image_size // 32)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        z, log_q_z = self.encode(inputs)
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z)
        outputs = self.decode(z)
        return [inputs, outputs, log_p_z, log_q_z]

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        input = inputs[0]
        output = inputs[1]
        log_p_z = inputs[2]
        log_q_z = inputs[3]
        log_p_x = -F.mse_loss(input, output, size_average=False) / input.size(0)

        elbo = log_p_x + log_p_z.sum(-1).mean() - log_q_z.mean()
        return {"loss": -elbo, "reconstruction_loss": -log_p_x}
