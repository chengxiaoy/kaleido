import torch
from torch import optim
from model.dc_gan import DCGAN
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision import datasets
from torch.utils.data import DataLoader


class DCGANExperiment(pl.LightningModule):

    def __init__(self,
                 gan_model: DCGAN,
                 params: dict) -> None:
        super(DCGANExperiment, self).__init__()

        self.model = gan_model
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
            return {'loss': train_loss['g_loss']}
        else:
            return {"loss": train_loss['d_loss']}

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

        optimizerG = optim.Adam(self.model.generator.parameters(), lr=self.params['LR'],
                                betas=(self.params['beta1'], 0.999))
        optims.append(optimizerG)

        optimizerD = optim.Adam(self.model.discriminator.parameters(), lr=self.params['LR'],
                                betas=(self.params['beta1'], 0.999))
        optims.append(optimizerD)

        return optims

    def train_dataloader(self):

        dataset = datasets.ImageFolder(root=self.params['data_path'],
                                       transform=transforms.Compose([
                                           transforms.Resize(self.params['image_size']),
                                           transforms.CenterCrop(self.params['image_size']),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        return torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True,
                                           num_workers=self.params['work_num'], drop_last=True)

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        dataset = datasets.ImageFolder(root=self.params['data_path'],
                                       transform=transforms.Compose([
                                           transforms.Resize(self.params['image_size']),
                                           transforms.CenterCrop(self.params['image_size']),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        return torch.utils.data.DataLoader(dataset, batch_size=self.params['sample_batch_size'], shuffle=True,
                                           num_workers=self.params['work_num'], drop_last=True)
