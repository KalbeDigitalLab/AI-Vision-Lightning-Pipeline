import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.train import train
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = 'src/train.py'
overrides = ['logger=[]']


def test_train_bodypartxr_fast_dev_run(cfg_bodyresnet18):
    """Run for 1 train, val and test step."""

    DATA_PATH = 'hub://jasonadrianzoom/bodyclassification'

    HydraConfig().set_config(cfg_bodyresnet18)
    with open_dict(cfg_bodyresnet18):
        cfg_bodyresnet18.paths.data_dir = DATA_PATH
        # cfg_bodyresnet18.paths.val_dir = DATA_PATH
        # cfg_bodyresnet18.paths.test_dir = DATA_PATH
        cfg_bodyresnet18.trainer.fast_dev_run = True
        cfg_bodyresnet18.trainer.accelerator = 'cpu'
    train(cfg_bodyresnet18)
