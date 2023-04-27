import os

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.dummy_dataset import vindr_2views_mammography_dataset_dir
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = 'src/train.py'
overrides = ['logger=[]']


@pytest.fixture
def cfg_phcresnet_global() -> DictConfig:
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='train.yaml',
                      return_hydra_config=True, overrides=['model=phcresnet_2view', 'data=vindrmammo'])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.input_size = [300, 250]
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope='function')
def cfg_phcresnet18(cfg_phcresnet_global, tmp_path) -> DictConfig:
    cfg = cfg_phcresnet_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.num_views = 2
        cfg.model.net._target_ = 'src.models.components.hypercomplex_models.PHCResNet18'

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope='function')
def cfg_phcresnet50(cfg_phcresnet_global, tmp_path) -> DictConfig:
    cfg = cfg_phcresnet_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.num_views = 2
        cfg.model._target_ = 'src.models.components.hypercomplex_models.PHCResNet50'

    yield cfg

    GlobalHydra.instance().clear()


def test_train_fast_dev_run(cfg_phcresnet18, vindr_2views_mammography_dataset_dir):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = vindr_2views_mammography_dataset_dir
        cfg_phcresnet18.trainer.fast_dev_run = True
        cfg_phcresnet18.trainer.accelerator = 'cpu'
    train(cfg_phcresnet18)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_phcresnet50, vindr_2views_mammography_dataset_dir):
    """Run for 1 train, val and test step on GPU."""
    HydraConfig().set_config(cfg_phcresnet50)
    with open_dict(cfg_phcresnet50):
        cfg_phcresnet50.paths.data_dir = vindr_2views_mammography_dataset_dir
        cfg_phcresnet50.trainer.fast_dev_run = True
        cfg_phcresnet50.trainer.accelerator = 'gpu'
    train(cfg_phcresnet50)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_phcresnet18, vindr_2views_mammography_dataset_dir):
    """Train 1 epoch on GPU with mixed-precision."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = vindr_2views_mammography_dataset_dir
        cfg_phcresnet18.trainer.max_epochs = 1
        cfg_phcresnet18.trainer.accelerator = 'cpu'
        cfg_phcresnet18.trainer.precision = 16
    train(cfg_phcresnet18)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_phcresnet18, vindr_2views_mammography_dataset_dir):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = vindr_2views_mammography_dataset_dir
        cfg_phcresnet18.trainer.max_epochs = 1
        cfg_phcresnet18.trainer.val_check_interval = 0.5
    train(cfg_phcresnet18)


@pytest.mark.slow
def test_train_ddp_sim(cfg_phcresnet18, vindr_2views_mammography_dataset_dir):
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes."""
    HydraConfig().set_config(cfg_phcresnet18)
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.paths.data_dir = vindr_2views_mammography_dataset_dir
        cfg_phcresnet18.data.batch_size = 2
        cfg_phcresnet18.trainer.max_epochs = 2
        cfg_phcresnet18.trainer.accelerator = 'cpu'
        cfg_phcresnet18.trainer.devices = 2
        cfg_phcresnet18.trainer.strategy = 'ddp_spawn'
        cfg_phcresnet18.trainer.fast_dev_run = True
    train(cfg_phcresnet18)


@pytest.mark.slow
def test_train_resume(tmp_path, cfg_phcresnet18, vindr_2views_mammography_dataset_dir):
    """Run 1 epoch, finish, and resume for another epoch."""
    with open_dict(cfg_phcresnet18):
        cfg_phcresnet18.trainer.max_epochs = 1
        cfg_phcresnet18.paths.data_dir = vindr_2views_mammography_dataset_dir

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
def test_experiments(tmp_path, vindr_2views_mammography_dataset_dir):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        startfile,
        '-m',
        'experiment=glob(vindrmammo_*)',
        'hydra.sweep.dir=' + str(tmp_path),
        'paths.data_dir=' + vindr_2views_mammography_dataset_dir,
        '++trainer.fast_dev_run=true',
        '++data.input_size=[300,250]',
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path, vindr_2views_mammography_dataset_dir):
    """Test default hydra sweep."""
    command = [
        startfile,
        '-m',
        'hydra.sweep.dir=' + str(tmp_path),
        'model=phcresnet_2view',
        'data=vindrmammo',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'model.lr=0.005,0.01',
        'paths.data_dir=' + vindr_2views_mammography_dataset_dir,
        '++trainer.fast_dev_run=true',
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.skip('Unknown error due to optimizer state on ddp sim.')
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path, vindr_2views_mammography_dataset_dir):
    """Test default hydra sweep with ddp sim."""
    command = [
        startfile,
        '-m',
        'hydra.sweep.dir=' + str(tmp_path),
        'model=phcresnet_2view',
        'data=vindrmammo',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'trainer=ddp_sim',
        'trainer.max_epochs=3',
        'model.lr=0.005,0.01,0.02',
        'paths.data_dir=' + vindr_2views_mammography_dataset_dir,
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path, vindr_2views_mammography_dataset_dir):
    """Test optuna sweep."""
    command = [
        startfile,
        '-m',
        'model=phcresnet_2view',
        'data=vindrmammo',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'hparams_search=phcresnet_optuna',
        'hydra.sweep.dir=' + str(tmp_path),
        'hydra.sweeper.n_trials=10',
        'hydra.sweeper.sampler.n_startup_trials=5',
        '++trainer.fast_dev_run=true',
        'paths.data_dir=' + vindr_2views_mammography_dataset_dir,
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.skip('Unknown error due to optimizer state on ddp sim.')
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path, vindr_2views_mammography_dataset_dir):
    """Test optuna sweep with wandb and ddp sim."""
    command = [
        startfile,
        '-m',
        'model=phcresnet_2view',
        'data=vindrmammo',
        'data.num_views=2',
        'data.input_size=[300,250]',
        'hparams_search=phcresnet_optuna',
        'hydra.sweep.dir=' + str(tmp_path),
        'hydra.sweeper.n_trials=5',
        'trainer=ddp_sim',
        'trainer.max_epochs=3',
        'logger=wandb',
        '++logger.anonymous=true',
        'paths.data_dir=' + vindr_2views_mammography_dataset_dir,
    ]
    run_sh_command(command)
