from model import BaseVAE, EncoderBottleneck, DecoderBottleneck
from model.types_ import Tensor
from torch import nn
import torch
import torch.nn.functional as F

from typing import Any, List
from model.vq_vae import VectorQuantizer, ResidualLayer


class ConditionalEncoder(nn.Module):

    def __init__(self, dim_num1, dim_num2, embedding_dim):
        super(ConditionalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(dim_num1, dim_num1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(dim_num2, dim_num2, kernel_size=4, stride=2, padding=1)

        self.conv = nn.Conv2d(dim_num2 + dim_num1, dim_num1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

        self.res_layer = nn.Sequential(
            ResidualLayer(dim_num1, dim_num1),
            ResidualLayer(dim_num1, dim_num1),
            ResidualLayer(dim_num1, dim_num1),
            nn.LeakyReLU(),
            nn.Conv2d(dim_num1, embedding_dim, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )

    def forward(self, input1, input2) -> Tensor:
        input1 = self.conv1(input1)
        input2 = self.conv2(input2)
        output = torch.cat([input1, input2], dim=1)
        output = self.conv(output)
        output = self.relu(output)
        output = self.res_layer(output)
        return output


class VQ_VAE2(BaseVAE):

    def __init__(self, **params):
        super(VQ_VAE2, self).__init__()

        self.beta = params['beta']
        self.in_channel = params['in_channels']
        self.num_embeddings = params['num_embeddings']
        self.embedding_dim = params['embedding_dim']

        # for top encoder we downsample the image by a factor of 8
        self.enc_res_layer = nn.Sequential(
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, self.embedding_dim, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.top_encoder = nn.Sequential(
            EncoderBottleneck(self.in_channel, 64, 4, 2, 1),
            EncoderBottleneck(64, 128, 4, 2, 1),
            EncoderBottleneck(128, 256, 4, 2, 1),
            EncoderBottleneck(256, 256, 3, 1, 1),
            self.enc_res_layer
        )

        self.top_vector_quantizer = VectorQuantizer(self.num_embeddings,
                                                    self.embedding_dim,
                                                    self.beta)

        # for bottom encoder we downsample the image by a factor of 4

        self.enc_res_layer_1 = nn.Sequential(
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            ResidualLayer(256, 256),
            nn.LeakyReLU(),
        )
        # receive image as input
        self.bottom_encoder_x = nn.Sequential(
            EncoderBottleneck(self.in_channel, 64, 4, 2, 1),
            EncoderBottleneck(64, 128, 4, 2, 1),
            EncoderBottleneck(128, 256, 3, 1, 1),
            self.enc_res_layer_1
        )

        self.bottom_encoder = ConditionalEncoder(256, self.embedding_dim, self.embedding_dim)

        self.bottom_vector_quantizer = VectorQuantizer(self.num_embeddings,
                                                       self.embedding_dim,
                                                       self.beta)

        self.dec_res_layer_1 = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            ResidualLayer(128, 128),
            nn.LeakyReLU()
        )
        # receive the top vector_quantizer as input
        self.decoder_top = nn.Sequential(
            self.dec_res_layer_1,
            DecoderBottleneck(128, 128, 4, 2, 1, 0)
        )

        self.dec_res_layer_2 = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            ResidualLayer(128, 128),
            nn.LeakyReLU()
        )
        # receive the top vector_quantizer as input
        self.decoder_bottom = nn.Sequential(
            self.dec_res_layer_2,
            DecoderBottleneck(128, 128, 3, 1, 1, 0)
        )

        self.dec_res_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
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
        top_encode = self.top_encoder(input)
        top_vector_q, top_loss = self.top_vector_quantizer(top_encode)

        bottom_mid = self.bottom_encoder_x(input)
        bottom_encode = self.bottom_encoder(bottom_mid, top_vector_q)
        bottom_vector_q, bottom_loss = self.bottom_vector_quantizer(bottom_encode)
        return [top_vector_q, bottom_vector_q, top_loss, bottom_loss]

    def decode(self, input) -> Any:
        top_vector_q, bottom_vector_q = input
        top = self.decoder_top(top_vector_q)
        bottom = self.decoder_bottom(bottom_vector_q)
        out_put = torch.cat([top, bottom], dim=1)
        out_put = self.decoder(out_put)
        return out_put

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[3]

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        top_vector_q, bottom_vector_q, top_loss, bottom_loss = self.encode(inputs)
        recons = self.decode([top_vector_q, bottom_vector_q])
        recons_loss = F.mse_loss(recons, inputs)
        return recons_loss, top_loss, bottom_loss, recons

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        recons_loss = inputs[0]
        top_vq_loss = inputs[1]
        bottom_vq_loss = inputs[2]
        return {"loss": recons_loss + top_vq_loss + bottom_vq_loss, "reconstruction_loss": recons_loss}
