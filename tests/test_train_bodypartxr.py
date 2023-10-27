import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.train import train
from tests.dummy_dataset import (
    bodypartxr_dummy_test_dataset_dir,
    bodypartxr_dummy_train_dataset_dir,
    bodypartxr_dummy_val_dataset_dir,
)
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = 'src/train.py'
overrides = ['logger=[]']


def test_train_bodypartxr_fast_dev_run(cfg_bodyresnet18, bodypartxr_dummy_test_dataset_dir, bodypartxr_dummy_train_dataset_dir, bodypartxr_dummy_val_dataset_dir):
    """Training Pipeline for Body Part Classification. Run for 1 train, val and test step using
    fast_dev_run.

    Args:
        cfg_bodyresnet18: Hydra experiment configuration
        bodypartxr_dummy_test_dataset_dir: Activeloop train dataset directory
        bodypartxr_dummy_train_dataset_dir: Activeloop val dataset directory
        bodypartxr_dummy_val_dataset_dir: Activeloop test dataset directory
    """
    HydraConfig().set_config(cfg_bodyresnet18)
    with open_dict(cfg_bodyresnet18):
        cfg_bodyresnet18.paths.train_dir = bodypartxr_dummy_train_dataset_dir
        cfg_bodyresnet18.paths.val_dir = bodypartxr_dummy_val_dataset_dir
        cfg_bodyresnet18.paths.test_dir = bodypartxr_dummy_test_dataset_dir
        cfg_bodyresnet18.trainer.fast_dev_run = True
        cfg_bodyresnet18.trainer.accelerator = 'cpu'
    train(cfg_bodyresnet18)
