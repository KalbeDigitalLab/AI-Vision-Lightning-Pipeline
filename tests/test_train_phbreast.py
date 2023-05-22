import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.train import train
from tests.dummy_dataset import (
    inbreast_2views_patches_dataset_path,
    inbreast_2views_single_dataset_path,
    inbreast_4views_multi_dataset_dir,
)
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = 'src/train.py'
overrides = ['logger=[]']


def test_train_phcresnet_binary_fast_dev_run(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.trainer.fast_dev_run = True
        cfg_phcresnet18.trainer.accelerator = 'cpu'
    train(cfg_phcresnet18)


def test_train_phcresnet_multiclass_fast_dev_run(cfg_phcresnet18, inbreast_2views_patches_dataset_path):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_patches_dataset_path
        cfg_phcresnet18.trainer.fast_dev_run = True
        cfg_phcresnet18.trainer.accelerator = 'cpu'
        cfg_phcresnet18.data.num_classes = 5
        cfg_phcresnet18.model.num_classes = 5
        cfg_phcresnet18.model.task = 'multiclass'
        cfg_phcresnet18.model.auto_lr = True
        cfg_phcresnet18.model.lr = 1.0
    train(cfg_phcresnet18)


def test_train_physbonet_binary_fast_dev_run_gpu(cfg_physbonet, inbreast_4views_multi_dataset_dir):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_physbonet)
    with open_dict(cfg_physbonet):
        cfg_physbonet.paths.data_dir = inbreast_4views_multi_dataset_dir
        cfg_physbonet.trainer.fast_dev_run = True
        cfg_physbonet.trainer.accelerator = 'cpu'
        cfg_physbonet.data.num_classes = 1
        cfg_physbonet.data.num_views = 4
        cfg_physbonet.model.num_classes = 1
        cfg_physbonet.model.task = 'binary'
    train(cfg_physbonet)


def test_train_physenet_binary_fast_dev_run_gpu(cfg_physenet, inbreast_4views_multi_dataset_dir):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_physenet)
    with open_dict(cfg_physenet):
        cfg_physenet.paths.data_dir = inbreast_4views_multi_dataset_dir
        cfg_physenet.trainer.fast_dev_run = True
        cfg_physenet.trainer.accelerator = 'cpu'
        cfg_physenet.data.num_classes = 1
        cfg_physenet.data.num_views = 4
        cfg_physenet.model.num_classes = 1
        cfg_physenet.model.task = 'binary'
    train(cfg_physenet)


def test_train_sch_none_fast_dev_run_gpu(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.trainer.fast_dev_run = True
        cfg_phcresnet18.trainer.accelerator = 'gpu'
        cfg_phcresnet18.model.scheduler_type = None
    train(cfg_phcresnet18)


def test_train_sch_lambda_fast_dev_run_gpu(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.trainer.fast_dev_run = True
        cfg_phcresnet18.trainer.accelerator = 'gpu'
        cfg_phcresnet18.model.scheduler_type = 'lambda'
    train(cfg_phcresnet18)


def test_train_sch_cosine_fast_dev_run_gpu(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.trainer.fast_dev_run = True
        cfg_phcresnet18.trainer.accelerator = 'gpu'
        cfg_phcresnet18.model.scheduler_type = 'cosine'
    train(cfg_phcresnet18)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_phcresnet50, inbreast_2views_single_dataset_path):
    """Run for 1 train, val and test step on GPU."""
    HydraConfig().set_config(cfg_phcresnet50)
    with open_dict(cfg_phcresnet50):
        cfg_phcresnet50.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet50.trainer.fast_dev_run = True
        cfg_phcresnet50.trainer.accelerator = 'gpu'
        cfg_phcresnet50.model.task = 'binary'
        cfg_phcresnet50.data.batch_size = 2
    train(cfg_phcresnet50)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Train 1 epoch on GPU with mixed-precision."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.trainer.max_epochs = 1
        cfg_phcresnet18.trainer.accelerator = 'gpu'
        cfg_phcresnet18.trainer.precision = 16
        cfg_phcresnet18.data.batch_size = 2
    train(cfg_phcresnet18)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.trainer.max_epochs = 1
        cfg_phcresnet18.trainer.val_check_interval = 0.5
    train(cfg_phcresnet18)


@pytest.mark.slow
def test_train_ddp_sim(cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path
        cfg_phcresnet18.data.batch_size = 2
        cfg_phcresnet18.trainer.max_epochs = 2
        cfg_phcresnet18.trainer.accelerator = 'cpu'
        cfg_phcresnet18.trainer.devices = 2
        cfg_phcresnet18.trainer.strategy = 'ddp_spawn'
        cfg_phcresnet18.trainer.fast_dev_run = True
    train(cfg_phcresnet18)


@pytest.mark.slow
def test_train_resume(tmp_path, cfg_phcresnet18, inbreast_2views_single_dataset_path):
    """Run 1 epoch, finish, and resume for another epoch."""
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.trainer.max_epochs = 1
        cfg_phcresnet18.paths.data_dir = inbreast_2views_single_dataset_path

    HydraConfig().set_config(cfg_phcresnet18)
    metric_dict_1, _ = train(cfg_phcresnet18)

    files = os.listdir(tmp_path / 'checkpoints')
    assert 'last.ckpt' in files
    assert 'epoch_000.ckpt' in files

    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.ckpt_path = str(tmp_path / 'checkpoints' / 'last.ckpt')
        cfg_phcresnet18.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_phcresnet18)

    files = os.listdir(tmp_path / 'checkpoints')
    assert 'epoch_000.ckpt' in files
    assert 'epoch_001.ckpt' not in files


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path, inbreast_2views_single_dataset_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        startfile,
        '-m',
        'experiment=phbreast_inbreast_phcresnet18_2views',
        'hydra.sweep.dir=' + str(tmp_path),
        'paths.data_dir=' + inbreast_2views_single_dataset_path,
        '++trainer.fast_dev_run=true',
        '++data.input_size=[300,250]',
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path, inbreast_2views_single_dataset_path):
    """Test default hydra sweep."""
    command = [
        startfile,
        '-m',
        'hydra.sweep.dir=' + str(tmp_path),
        'model=phcresnet',
        'data=phbreast',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'model.lr=0.005,0.01',
        'paths.data_dir=' + inbreast_2views_single_dataset_path,
        '++trainer.fast_dev_run=true',
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.skip('Unknown error due to optimizer state on ddp sim.')
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path, inbreast_2views_single_dataset_path):
    """Test default hydra sweep with ddp sim."""
    command = [
        startfile,
        '-m',
        'hydra.sweep.dir=' + str(tmp_path),
        'model=phcresnet',
        'data=phbreast',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'trainer=ddp_sim',
        'trainer.max_epochs=3',
        'model.lr=0.005,0.01,0.02',
        'paths.data_dir=' + inbreast_2views_single_dataset_path,
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path, inbreast_2views_single_dataset_path):
    """Test optuna sweep."""
    command = [
        startfile,
        '-m',
        'model=phcresnet',
        'data=phbreast',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'hparams_search=phcresnet_optuna',
        'hydra.sweep.dir=' + str(tmp_path),
        'hydra.sweeper.n_trials=10',
        'hydra.sweeper.sampler.n_startup_trials=5',
        '++trainer.fast_dev_run=true',
        'paths.data_dir=' + inbreast_2views_single_dataset_path,
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.skip('Unknown error due to optimizer state on ddp sim.')
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path, inbreast_2views_single_dataset_path):
    """Test optuna sweep with wandb and ddp sim."""
    command = [
        startfile,
        '-m',
        'model=phcresnet',
        'data=phbreast',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'hparams_search=phcresnet_optuna',
        'hydra.sweep.dir=' + str(tmp_path),
        'hydra.sweeper.n_trials=5',
        'trainer=ddp_sim',
        'trainer.max_epochs=3',
        'logger=wandb',
        '++logger.anonymous=true',
        'paths.data_dir=' + inbreast_2views_single_dataset_path,
    ]
    run_sh_command(command)
