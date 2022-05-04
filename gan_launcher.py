import yaml
import argparse
import numpy as np

from model import *
from experiment import StyleGanExperiment, ALAE_Experiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
args.filename = "configs/alae.yaml"
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
experiment = ALAE_Experiment(model, config['exp_params'])

checkpoint_callback = ModelCheckpoint(dirpath="./logs/ALAE", every_n_train_steps=1)

runner = Trainer(callbacks=[checkpoint_callback],
                 min_epochs=1,
                 logger=tb_logger,
                 log_every_n_steps=100,
                 val_check_interval=1,
                 num_sanity_val_steps=3,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
