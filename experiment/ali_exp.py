import torch
from torch.optim.rmsprop import RMSprop
from model.ali import ALI
import pytorch_lightning as pl
import torchvision.utils as vutils
from data_samples import get_celebA_dataloader


class ALIExperiment(pl.LightningModule):

    def __init__(self,
                 ali: ALI,
                 params: dict) -> None:
        super(ALIExperiment, self).__init__()

        self.model = ali
        self.params = params
        self.curr_device = None

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              batch_size=self.params['batch_size'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        if optimizer_idx == 0:
            return {"loss": train_loss["gan_d_loss"]}

        if optimizer_idx == 1:
            return {"loss": train_loss["gan_g_loss"]}


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
                              nrow=6)
        except:
            pass

        del test_input, recons  # , samples

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        self.model.zero_grad()

    def configure_optimizers(self):

        optims = []
        scheds = []



        dis_optimizer = RMSprop(self.model.discriminator.parameters(), lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'])
        optims.append(dis_optimizer)

        net_optimizer = RMSprop([{'params': self.model.encoder.parameters()},
                                 {'params': self.model.fc_mu.parameters()},
                                 {'params': self.model.fc_var.parameters()},
                                 {"params": self.model.decoder_input.parameters()},
                                 {"params": self.model.decoder.parameters()},
                                 {"params": self.model.final_layer.parameters()}
                                 ],
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'])
        optims.append(net_optimizer)
        return optims, scheds

    @pl.data_loader
    def train_dataloader(self):
        return get_celebA_dataloader(self.params["image_size"], self.params["batch_size"], split="train")

    @pl.data_loader
    def val_dataloader(self):
        return get_celebA_dataloader(self.params["image_size"], self.params["batch_size"], split="valid")

    def test_dataloader(self):
        return get_celebA_dataloader(self.params["image_size"], self.params["sample_batch_size"], split="test")
