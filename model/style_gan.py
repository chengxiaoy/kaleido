import torch
from torch import nn
from torch.autograd import grad

from model.style_gan_util import StyledGenerator, Discriminator, F
import random


class StyleGAN(nn.Module):
    def __init__(self, **params):
        super(StyleGAN, self).__init__()
        self.code_size = params['code_dim']
        self.generator = StyledGenerator(code_dim=self.code_size)
        self.discriminator = Discriminator()
        self.loss_type = params['loss']
        self.mixing = params['mixing']

    def forward(self, real_image, step, alpha, train_dis=True):
        batch_size = real_image.size(0)

        # for fake image
        # get latent code
        if self.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, batch_size, self.code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        else:
            gen_in1, gen_in2 = torch.randn(2, batch_size, self.code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)
        # for discriminator part
        if train_dis:

            # discriminator real image
            if self.loss_type == 'wgan-gp':
                real_predict = self.discriminator(real_image, step=step, alpha=alpha)
                real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
                dis_real_loss = -real_predict

            elif self.loss_type == 'r1':
                real_image.requires_grad = True
                real_scores = self.discriminator(real_image, step=step, alpha=alpha)
                real_predict = F.softplus(-real_scores).mean()
                dis_real_loss = real_predict
                # real_predict.backward(retain_graph=True)

                grad_real = grad(
                    outputs=real_scores.sum(), inputs=real_image, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                grad_penalty = 10 / 2 * grad_penalty
                # grad_penalty.backward()

            fake_image = self.generator(gen_in1, step=step, alpha=alpha)
            fake_predict = self.discriminator(fake_image, step=step, alpha=alpha)

            if self.loss_type == 'wgan-gp':
                fake_predict = fake_predict.mean()
                dis_fake_loss = fake_predict
                # fake_predict.backward()

                eps = torch.rand(batch_size, 1, 1, 1).cuda()
                x_hat = eps * real_image.data + (1 - eps) * fake_image.data
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

            return dis_fake_loss + dis_real_loss + grad_penalty, torch.Tensor([0.0]).cuda()
        # for generator part
        fake_image = self.generator(gen_in2, step=step, alpha=alpha)
        predict = self.discriminator(fake_image, step=step, alpha=alpha)

        if self.loss_type == 'wgan-gp':
            generator_loss = -predict.mean()

        elif self.loss_type == 'r1':
            generator_loss = F.softplus(-predict).mean()

        return torch.Tensor([0.0]).cuda(), generator_loss
