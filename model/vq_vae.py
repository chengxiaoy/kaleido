from model import BaseVAE, EncoderBottleneck, DecoderBottleneck
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F

from typing import Any, List


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


class VQ_VAE(BaseVAE):
    def __init__(self, **params):
        super(VQ_VAE, self).__init__()

        self.in_channel = params["in_channels"]
        self.num_embeddings = params["num_embeddings"]
        self.embedding_dim = params["embedding_dim"]
        self.beta = params["beta"]

        self.enc_res_layer = nn.Sequential(
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, self.embedding_dim, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.encoder = nn.Sequential(
            EncoderBottleneck(self.in_channel, 128, 4, 2, 1),
            EncoderBottleneck(128, 256, 4, 2, 1),
            EncoderBottleneck(256, 256, 3, 1, 1),
            self.enc_res_layer
        )

        self.vq_layer = VectorQuantizer(self.num_embeddings,
                                        self.embedding_dim,
                                        self.beta)

        self.dec_res_layer = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            self.dec_res_layer,
            DecoderBottleneck(256, 128, 4, 2, 1, 0),
            nn.ConvTranspose2d(128, out_channels=self.in_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        encoding = self.encoder(input)

        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [quantized_inputs, vq_loss]

    def decode(self, input: Tensor) -> Any:
        recons = self.decoder(input)
        return recons

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[2]

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        quantized_inputs, vq_loss = self.encode(inputs)
        recons = self.decode(quantized_inputs)
        recons_loss = F.mse_loss(recons, inputs)
        return recons_loss, vq_loss, recons

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        recons_loss = inputs[0]
        vq_loss = inputs[1]
        return {"loss": recons_loss + vq_loss, "reconstruction_loss": recons_loss}
