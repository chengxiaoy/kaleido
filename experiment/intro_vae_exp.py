import torch
from torch.optim.adam import Adam
from model.intro_vae import INTRO_VAE
import pytorch_lightning as pl
import torchvision.utils as vutils
from data_samples import get_celebA_dataloader, get_ffhq_dataloader


class INTRO_VAEExperiment(pl.LightningModule):

    def __init__(self,
                 intro_vae: INTRO_VAE,
                 params: dict) -> None:
        super(INTRO_VAEExperiment, self).__init__()

        self.model = intro_vae
        self.params = params
        self.curr_device = None

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        if self.current_epoch < 2:
            self.model.alpha = 0.0
        else:
            self.model.alpha = 0.25

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              batch_size=self.params['batch_size'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        if optimizer_idx == 0:
            return {"loss": train_loss["decoder_loss"]}

        if optimizer_idx == 1:
            return {"loss": train_loss["encoder_loss"]}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            batch_size=self.params['batch_size'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):

        optimizer.step()
        self.model.zero_grad()

        # if optimizer_idx == 0:
        #     optimizer.step()
        #     self.model.zero_grad()
        # if optimizer_idx == 1:
        #     if batch_idx % 2 == 0:
        #         optimizer.step()
        #         self.model.zero_grad()

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
        if self.params['dataset'] == 'ffhq':
            return get_ffhq_dataloader(self.params['ffhq_data_path'], self.params["image_size"],
                                       self.params["batch_size"], split="train")
        return get_celebA_dataloader(self.params["image_size"], self.params["batch_size"], split="train")

    def val_dataloader(self):
        if self.params['dataset'] == 'ffhq':
            return get_ffhq_dataloader(self.params['ffhq_data_path'], self.params["image_size"],
                                       self.params["batch_size"], split="val")
        return get_celebA_dataloader(self.params["image_size"], self.params["batch_size"], split="valid")

    def test_dataloader(self):
        if self.params['dataset'] == 'ffhq':
            return get_ffhq_dataloader(self.params['ffhq_data_path'], self.params["image_size"],
                                       self.params['sample_batch_size'], split="val")
        return get_celebA_dataloader(self.params["image_size"], self.params['sample_batch_size'], split="test")
