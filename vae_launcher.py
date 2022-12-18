import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from experiment.soft_intro_vae_exp import SOFT_INTRO_VAEExperiment
from model import *

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
args.filename = "configs/soft_intro_vae.yaml"
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

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = SOFT_INTRO_VAEExperiment(model, config['exp_params'])

checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir, every_n_epochs=1)
early_stopping = EarlyStopping(mode="min", monitor='val_loss')

runner = Trainer(callbacks=[checkpoint_callback],
                 min_epochs=1,
                 logger=tb_logger,
                 log_every_n_steps=10,
                 # val_check_interval=1,
                 num_sanity_val_steps=3,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
