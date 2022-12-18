import os

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch.optim.adam import Adam

from data_samples import get_celebA_dataloader, get_ffhq_dataloader, get_beauty_dataloader
from model.soft_intro_vae import SOFT_INTRO_VAE


class SOFT_INTRO_VAEExperiment(pl.LightningModule):

    def __init__(self,
                 intro_vae: SOFT_INTRO_VAE,
                 params: dict) -> None:
        super(SOFT_INTRO_VAEExperiment, self).__init__()

        self.model = intro_vae
        self.params = params
        self.curr_device = None
        self.batch_size = params['batch_size']
        self.data_set = params['data_set']
        self.resolution = params['resolution']

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        real_batch = real_img.to(self.curr_device)
        scale = self.model.scale
        beta_rec = self.model.beta_rec
        beta_kl = self.model.beta_kl
        beta_neg = self.model.beta_neg
        gamma_r = self.model.gamma_r

        real_mu, real_logvar = self.model.encode(real_batch)
        z = self.model.reparameter_trick(real_mu, real_logvar)

        noise_batch = torch.randn(size=z.shape).to(self.curr_device)

        if optimizer_idx == 1:
            # =========== Update E ================
            rec = self.model.decode(z)
            fake = self.model.decode(noise_batch)

            loss_rec = self.model.calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")
            lossE_real_kl = self.model.calc_kl(real_logvar, real_mu, reduce="mean")

            rec_mu, rec_logvar, z_rec, rec_rec = self.model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = self.model(fake.detach())

            kl_rec = self.model.calc_kl(rec_logvar, rec_mu, reduce='none')
            kl_fake = self.model.calc_kl(fake_logvar, fake_mu, reduce='none')

            loss_rec_rec_e = self.model.calc_reconstruction_loss(rec, rec_rec, reduction='none')
            loss_rec_fake_e = self.model.calc_reconstruction_loss(fake, rec_fake, reduction='none')

            expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec)).exp().mean()
            expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake)).exp().mean()

            lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
            lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

            lossE = lossE_real + lossE_fake
            return {"loss": lossE}
        elif optimizer_idx == 0:
            # ========= Update D ==================
            fake = self.model.decode(noise_batch)
            rec = self.model.decode(z.detach())
            loss_rec = self.model.reconstruction_loss(real_batch, rec)

            rec_mu, rec_logvar = self.model.encode(rec)
            z_rec = self.model.reparameter_trick(rec_mu, rec_logvar)

            fake_mu, fake_logvar = self.model.encode(fake)
            z_fake = self.model.reparameter_trick(fake_mu, fake_logvar)

            rec_rec = self.model.decode(z_rec.detach())
            rec_fake = self.model.decode(z_fake.detach())

            loss_rec_rec = self.model.reconstruction_loss(rec.detach(), rec_rec)
            loss_fake_rec = self.model.reconstruction_loss(fake.detach(), rec_fake)

            lossD_rec_kl = self.model.kl_loss(rec_logvar, rec_mu)
            lossD_fake_kl = self.model.kl_loss(fake_logvar, fake_mu)

            lossD = scale * (loss_rec * beta_rec + (
                    lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                     loss_rec_rec + loss_fake_rec))
            return {"loss": lossD}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        pass

    def training_epoch_end(self, outputs) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        if not os.path.exists(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"):
            os.mkdir(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/")
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          range=(-1, 1),
                          nrow=6)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(36,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              range=(-1, 1),
                              nrow=6)
        except:
            pass

        del test_input, recons  # , samples

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # optimizer.step(closure=optimizer_closure)
        # self.model.zero_grad()

        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
        if optimizer_idx == 1:
            if batch_idx % 2 == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()

        # if optimizer_idx == 0:
        #     if batch_idx % 2 == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()
        #
        # # update discriminator opt every 4 steps
        # if optimizer_idx == 1:
        #     if batch_idx % 4 == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()

    def configure_optimizers(self):

        optims = []
        scheds = []
        g_optimizer = Adam([
            {"params": self.model.decoder_input.parameters()},
            {"params": self.model.decoder.parameters()},
            {"params": self.model.final_layer.parameters()}
        ],
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay'])
        optims.append(g_optimizer)

        d_optimizer = Adam([
            {"params": self.model.encoder.parameters()},
            {"params": self.model.fc_mu.parameters()},
            {"params": self.model.fc_var.parameters()}
        ], lr=self.params['LR'],
            weight_decay=self.params['weight_decay'])
        optims.append(d_optimizer)
        return optims, scheds

    def train_dataloader(self):
        if self.data_set == 'celeba':
            return get_celebA_dataloader(self.resolution, self.batch_size, split="train")
        elif self.data_set == 'ffhq':
            return get_ffhq_dataloader(self.resolution, self.batch_size)
        elif self.data_set == 'beauty':
            return get_beauty_dataloader(self.resolution, self.batch_size)

    def val_dataloader(self):
        if self.data_set == 'celeba':
            return get_celebA_dataloader(self.resolution, self.batch_size, split="val")
        elif self.data_set == 'ffhq':
            return get_ffhq_dataloader(self.resolution, self.batch_size, split="val")
        elif self.data_set == 'beauty':
            return get_beauty_dataloader(self.resolution, self.batch_size, split="val")

    def test_dataloader(self):
        if self.data_set == 'celeba':
            return get_celebA_dataloader(self.resolution, 36, split="val")
        elif self.data_set == 'ffhq':
            return get_ffhq_dataloader(self.resolution, 36, split="val")
        elif self.data_set == 'beauty':
            return get_beauty_dataloader(self.resolution, 36, split="val")
