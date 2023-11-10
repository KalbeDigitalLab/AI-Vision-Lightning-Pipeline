import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.train import train
from tests.bc_dummy_dataset import (
    deeplake_dummy_dataset_dir,
    deeplake_dummy_test_dataset_dir,
    deeplake_dummy_train_dataset_dir,
    deeplake_dummy_val_dataset_dir,
)
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = 'src/train.py'
overrides = ['logger[]']


def test_train_bcc_fast_dev_run(cfg_bcresnet18, deeplake_dummy_train_dataset_dir, deeplake_dummy_val_dataset_dir, deeplake_dummy_test_dataset_dir):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_bcresnet18)
    with open_dict(cfg_bcresnet18):
        cfg_bcresnet18.paths.train_dir = deeplake_dummy_train_dataset_dir
        cfg_bcresnet18.paths.val_dir = deeplake_dummy_val_dataset_dir
        cfg_bcresnet18.paths.test_dir = deeplake_dummy_test_dataset_dir
        cfg_bcresnet18.trainer.fast_dev_run = True
        cfg_bcresnet18.trainer.accelerator = 'cpu'
    train(cfg_bcresnet18)
