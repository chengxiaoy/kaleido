import math

import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import transforms
from model.style_gan import StyleGAN
from model.style_gan import StyledGenerator
from torchvision import utils
from torch.utils.data import Dataset, DataLoader
import lmdb
from PIL import Image
from io import BytesIO
from data_samples import get_ffhq_dataloader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class StyleGanExperiment(pl.LightningModule):

    def __init__(self, model: StyleGAN, params: dict):
        super(StyleGanExperiment, self).__init__()
        self.model = model
        self.code_size = params['code_dim']
        self.g_running = StyledGenerator(self.code_size)
        self.g_running.train(False)
        self.accumulate(self.g_running, self.model.generator, 0)

        self.resolution = params['resolution']
        self.step = int(math.log2(self.resolution)) - 2
        self.alpha = 1
        self.lr = params['LR']
        self.path = params['path']
        self.batch_size = params['batch_size']

    def forward(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        return get_ffhq_dataloader(self.path, self.resolution, self.batch_size)

    def training_step(self, inputs: torch.Tensor, batch_idx, optimizer_idx=0):
        imgs, labels = inputs
        if optimizer_idx == 0:
            requires_grad(self.model.generator, False)
            requires_grad(self.model.discriminator, True)
            dis_loss, gen_loss = self.model(imgs, self.step, self.alpha, train_dis=True)

        if optimizer_idx == 1:
            requires_grad(self.model.generator, True)
            requires_grad(self.model.discriminator, False)
            dis_loss, gen_loss = self.model(imgs, self.step, self.alpha, train_dis=False)

        # self.logger.experiment.log({key: val.item() for key, val in {"d_loss": dis_loss, "g_loss": gen_loss}.items()})

        if optimizer_idx == 0:
            return {"loss": dis_loss}
        if optimizer_idx == 1:
            return {"loss": gen_loss}

    def sample_data(self, epoch):
        # sample pic
        images = []
        gen_i, gen_j = (10, 5)
        with torch.no_grad():
            for _ in range(gen_i):
                images.append(
                    self.g_running(
                        torch.randn(gen_j, self.code_size).cuda(), step=self.step, alpha=self.alpha
                    ).data.cpu()
                )
        utils.save_image(
            torch.cat(images, 0),
            f'{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/epoch_{epoch}.png',
            nrow=gen_i,
            normalize=True,
            range=(-1, 1),
        )

    def call_back(self):
        self.sample_data(self.current_epoch)
        torch.save(
            self.g_running.state_dict(),
            f'{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/epoch_{self.current_epoch}.model'
        )

    def validation_step(self, *args, **kwargs):
        pass

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if optimizer_idx == 0:
            optimizer.step()
            self.model.zero_grad()
        elif optimizer_idx == 1:
            optimizer.step()
            self.accumulate(self.g_running, self.model.generator)
            if batch_idx == 1600:
                self.call_back()

    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def configure_optimizers(self):

        g_optimizer = optim.Adam(
            self.model.generator.generator.parameters(), lr=self.lr, betas=(0.0, 0.99)
        )
        g_optimizer.add_param_group(
            {
                'params': self.model.generator.style.parameters(),
                'lr': self.lr * 0.01,
            }
        )
        d_optimizer = optim.Adam(self.model.discriminator.parameters(), lr=self.lr, betas=(0.0, 0.99))

        return [d_optimizer, g_optimizer]


def load_model(model_cp_path):
    code_size = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = StyledGenerator(code_size).to(device=device).train(False)
    generator.load_state_dict(torch.load(model_cp_path))
    return generator


if __name__ == '__main__':
    generator = load_model('../logs/Style_GAN/version_2/epoch_6.model')
    mixing_range = (2, 4)
    images1 = []
    images2 = []
    images_mix = []
    gen_i, gen_j = (10, 5)
    with torch.no_grad():
        for _ in range(gen_i):
            input1 = torch.randn(gen_j, 512).cuda()
            images1.append(
                generator(
                    input1, step=int(math.log2(64)) - 2, alpha=1
                ).data.cpu()
            )

            input2 = torch.randn(gen_j, 512).cuda()
            images2.append(
                generator(
                    input2, step=int(math.log2(64)) - 2, alpha=1
                ).data.cpu()
            )

            images_mix.append(
                generator(
                    [input1, input2], step=int(math.log2(64)) - 2, alpha=1, mixing_range=mixing_range
                ).data.cpu()
            )

    utils.save_image(
        torch.cat(images1, 0),
        f'../logs/Style_GAN/sample1.png',
        nrow=gen_i,
        normalize=True,
        range=(-1, 1),
    )

    utils.save_image(
        torch.cat(images2, 0),
        f'../logs/Style_GAN/sample2.png',
        nrow=gen_i,
        normalize=True,
        range=(-1, 1),
    )

    utils.save_image(
        torch.cat(images_mix, 0),
        f'../logs/Style_GAN/sample_mix.png',
        nrow=gen_i,
        normalize=True,
        range=(-1, 1),
    )
