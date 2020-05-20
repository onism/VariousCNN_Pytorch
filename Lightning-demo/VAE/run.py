import yaml
import numpy as np
import os
import torch 
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger

from beta_vae import BetaVAE
from vae_wrapper import VAEWrapper

with open('config/betavae.yaml', 'r') as file:
    config = yaml.safe_load(file)

 
tt_logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    name='lightning_logs'
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])

model = BetaVAE(**config['model_params'])
wrapper = VAEWrapper(model, config['exp_params'])
runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(wrapper)

