from typing import Any, List

from model import BaseVAE, EncoderBottleneck, DecoderBottleneck
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

    def get_index(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        encoding_inds = encoding_inds.reshape(latents_shape[:-1] + (1,))
        return encoding_inds

    def get_embedding(self, indexs):
        indexs_size = indexs.size()  # BHW
        encoding_inds = indexs.view(-1, 1)
        # Convert to one-hot encodings
        device = encoding_inds.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(indexs_size + (self.D,))  # [B x H x W x D]

        return quantized_latents.permute(0, 3, 1, 2).contiguous()  # [B x D x H x W]


class BetaVAE(BaseVAE):

    def __init__(self, **params):
        # 将 VanillaVAE 类型的self 转换成为BaseVAE 并执行基类的init方法
        super(BetaVAE, self).__init__()
        self.image_size = params['image_size']
        self.in_channel = params["in_channels"]
        self.lantent_dim = params["latent_dim"]
        self.beta = params["beta"]

        # 将图片down sample 32 倍
        self.encoder = nn.Sequential(
            EncoderBottleneck(self.in_channel, 32, 3, 2, 1),
            EncoderBottleneck(32, 64, 3, 2, 1),
            EncoderBottleneck(64, 128, 3, 2, 1),
            EncoderBottleneck(128, 256, 3, 2, 1),
            EncoderBottleneck(256, 512, 3, 2, 1),
        )

        self.fc_mu = nn.Linear(512 * (self.image_size // 32) ** 2, self.lantent_dim)
        self.fc_var = nn.Linear(512 * (self.image_size // 32) ** 2, self.lantent_dim)

        self.decoder_input = nn.Linear(self.lantent_dim, 512 * (self.image_size // 32) ** 2)

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
        mu, logvar = self.encode(inputs)
        latent_code = self.reparameter_trick(mu, logvar)
        outputs = self.decode(latent_code)
        return [inputs, outputs, mu, logvar]

    def reparameter_trick(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(logvar)
        return mu + eps * std

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        input = inputs[0]
        output = inputs[1]
        mu = inputs[2]
        logvar = inputs[3]
        # prior regulation
        prior_loss = torch.mean(torch.sum(-0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)), dim=1), dim=0)
        # reconstruction_loss = F.mse_loss(input, output)
        reconstruction_loss = F.mse_loss(input, output, size_average=False) / input.size(0)

        loss = self.beta * prior_loss + reconstruction_loss
        return {"loss": loss, "reconstruction_loss": reconstruction_loss, "prior_loss": prior_loss}

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
