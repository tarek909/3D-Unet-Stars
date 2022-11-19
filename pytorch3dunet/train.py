import random

import torch

from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration



    # create trainer
    cfg=load_config()
    trainer = create_trainer(cfg)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
