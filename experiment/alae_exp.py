from typing import Optional

import numpy as np
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.optim.adam import Adam
from torch import optim
from model.alae import ALAE
import pytorch_lightning as pl
import torchvision.utils as vutils
from data_samples import get_celebA_dataloader, get_ffhq_dataloader
from torch.nn import functional as F
import random
from torch.autograd import grad
from torchvision import utils


class ALAE_Experiment(pl.LightningModule):

    def __init__(self, model: ALAE, params: dict) -> None:
        super(ALAE_Experiment, self).__init__()
        self.model = model
        self.code_size = params['code_dim']

        self.params = params
        self.resolution = params['resolution']
        self.step = int(np.log2(self.resolution)) - 2
        self.cur_device = None
        self.loss_type = params['loss']
        self.mixing = params['mixing']
        self.alpha = params['alpha']
        self.batch_size = params['batch_size']
        self.data_set = params['data_set']
        self.lr = params['LR']

    def train_dataloader(self):
        if self.data_set == 'celeba':
            return get_celebA_dataloader(self.resolution, self.batch_size, "train")
        elif self.data_set == 'ffhq':
            return get_ffhq_dataloader(self.resolution, self.batch_size)

    def training_step(self, batch, batch_idx, optimizer_idx) -> dict:
        real_img, labels = batch
        alpha = self.alpha
        self.cur_device = real_img.device
        batch_size = real_img.size(0)
        code_dim = self.params['code_dim']
        step = int(np.log2(self.resolution)) - 2

        if self.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, batch_size, code_dim, device=self.cur_device
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        else:
            gen_in1, gen_in2 = torch.randn(2, batch_size, code_dim, device=self.cur_device).chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        # train F & G
        if optimizer_idx == 1:
            origin_z = gen_in1

            w = self.model.trans(origin_z)
            x = self.model.generate([w], batch=batch_size, step=step)
            w_ = self.model.encode(x, step=step, alpha=alpha)
            # recipient loss in latent space
            # w_recons_loss = self.reconstruction_loss(w_, w, size_average=True)
            y = self.model.discriminate(w_)

            if self.loss_type == 'r1':
                generator_loss = F.softplus(-y).mean()
            if self.loss_type == 'wgan-gp':
                generator_loss = -y.mean()
            self.logger.log_metrics({"loss": generator_loss}, step=batch_idx)
            return {"loss": generator_loss}

        # train E & D
        if optimizer_idx == 0:
            origin_z = gen_in2

            # discriminator real image
            if self.loss_type == 'wgan-gp':
                real_predict = self.model.discriminate(self.model.encode(real_img, step=step, alpha=alpha))
                real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
                dis_real_loss = -real_predict

            elif self.loss_type == 'r1':
                real_img.requires_grad = True
                real_scores = self.model.discriminate(self.model.encode(real_img, step=step, alpha=alpha))
                real_predict = F.softplus(-real_scores).mean()
                dis_real_loss = real_predict
                # real_predict.backward(retain_graph=True)

                grad_real = grad(
                    outputs=real_scores.sum(), inputs=real_img, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                grad_penalty = 10 / 2 * grad_penalty
                # grad_penalty.backward()

            w = self.model.trans(origin_z)
            fake_image = self.model.generate([w], batch=batch_size, step=step)
            w_ = self.model.encode(fake_image, step=step, alpha=alpha)
            w_recons_loss = self.reconstruction_loss(w_, w, size_average=True)
            fake_predict = self.model.discriminate(w_)

            # fake_image = self.model.generate([self.model.trans(origin_z)], step=step, alpha=alpha, batch=batch_size)
            # fake_predict = self.model.discriminate(self.model.encode(fake_image, step=step, alpha=alpha))

            if self.loss_type == 'wgan-gp':
                fake_predict = fake_predict.mean()
                dis_fake_loss = fake_predict
                # fake_predict.backward()

                eps = torch.rand(batch_size, 1, 1, 1).cuda()
                x_hat = eps * real_img.data + (1 - eps) * fake_image.data
                x_hat.requires_grad = True
                hat_predict = self.discriminator(x_hat, step=step, alpha=alpha)

                grad_x_hat = grad(
                    outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
                )[0]

                grad_penalty = (
                        (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
                ).mean()
                grad_penalty = 10 * grad_penalty
                # grad_penalty.backward()

            if self.loss_type == "r1":
                fake_predict = F.softplus(fake_predict).mean()
                dis_fake_loss = fake_predict

            self.logger.log_metrics({"d_f_loss": dis_fake_loss,
                                     "d_r_loss": dis_real_loss,
                                     "d_p_loss": grad_penalty,
                                     "w_recons_loss": w_recons_loss
                                     }, step=batch_idx)
            return {"loss": dis_fake_loss + dis_real_loss + grad_penalty + w_recons_loss}

        # train G & E
        if optimizer_idx == 2:
            origin_z = gen_in2
            w = self.model.trans(origin_z)
            fake_image = self.model.generate([w], batch=batch_size, step=step)
            w_ = self.model.encode(fake_image, step=step, alpha=alpha)
            # recipient loss in latent space
            w_recons_loss = self.reconstruction_loss(w_, w, size_average=True)

            self.logger.log_metrics({"w_recons_loss": w_recons_loss}, step=batch_idx)
            return {"loss": w_recons_loss}

    def sample_data(self, epoch):
        # sample pic
        images = []
        gen_i, gen_j = (10, 5)
        with torch.no_grad():
            for _ in range(gen_i):
                images.append(
                    self.model.generate(
                        [self.model.trans(torch.randn(gen_j, self.code_size).cuda())], batch=gen_j, step=self.step,
                        alpha=self.alpha
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

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.sample_data(self.current_epoch)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # update discriminator opt every step
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        # update generator opt every 2 steps
        elif optimizer_idx == 1:
            # if (batch_idx + 1) % 2 == 0:
            optimizer.step(closure=optimizer_closure)


    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def configure_optimizers(self):

        d_optimizer = optim.Adam([{
            "params": self.model.discriminator.parameters()
        }, {
            "params": self.model.encoder.parameters()
        }
        ], lr=self.lr, betas=(0.0, 0.99))

        g_optimizer = optim.Adam(
            self.model.generator.parameters(), lr=self.lr, betas=(0.0, 0.99)
        )
        g_optimizer.add_param_group(
            {
                'params': self.model.t.parameters(),
                'lr': self.lr * 0.01,
            }
        )
        ae_optimizer = optim.Adam([{
            "params": self.model.encoder.parameters()
        }
        ], lr=self.lr, betas=(0.0, 0.99))
        return [d_optimizer, g_optimizer]

    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        return v_kl.mean()

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.mean(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error
