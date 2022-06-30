import torch
import yaml
import argparse
import numpy as np

from model import *
from experiment import StyleGanExperiment, ALAE_Experiment, StyleGAN2_ADA_Experiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
args.filename = "configs/stylegan2_ada.yaml"
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])

np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

# model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = StyleGAN2_ADA_Experiment(config['exp_params'])
# experiment = ALAE_Experiment.load_from_checkpoint("logs/ALAE/version_15/epoch=4-step=14624.ckpt", model=model,
#                                                   params=config['exp_params'])

checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir, every_n_epochs=1)

runner = Trainer(callbacks=[checkpoint_callback],
                 min_epochs=1,
                 logger=tb_logger,
                 log_every_n_steps=10,
                 val_check_interval=1,
                 num_sanity_val_steps=3,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment,ckpt_path="logs/StyleGAN2_ADA/version_0/epoch=68-step=69276.ckpt")
# runner.fit(experiment, ckpt_path="logs/ALAE/version_20/epoch=111-step=28111.ckpt")
