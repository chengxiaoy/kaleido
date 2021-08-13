from .types_ import *
from torch import nn
from abc import abstractmethod


class EncoderBottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(EncoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, output_padding):
        super(DecoderBottleneck, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.upsample(x)
        output = self.bn(output)
        output = self.relu(output)
        return output


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
