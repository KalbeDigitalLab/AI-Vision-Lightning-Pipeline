"""This file prepares config fixtures for other tests."""

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope='package')
def cfg_train_global() -> DictConfig:
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='train.yaml',
                      return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.batch_size = 4
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope='package')
def cfg_eval_global() -> DictConfig:
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='eval.yaml',
                      return_hydra_config=True, overrides=['ckpt_path=.'])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.batch_size = 4
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope='function')
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope='function')
def cfg_eval(cfg_eval_global, tmp_path) -> DictConfig:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope='package')
def cfg_phcresnet_global() -> DictConfig:
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='train.yaml',
                      return_hydra_config=True, overrides=['model=phcresnet', 'data=phbreast'])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.input_size = [300, 250]
            cfg.data.num_workers = 0
            cfg.data.batch_size = 4
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
        cfg.model.net._target_ = 'src.models.components.hypercomplex_models.PHCResNet50'

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope='package')
def cfg_physbonet_global() -> DictConfig:
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='train.yaml',
                      return_hydra_config=True, overrides=['model=physbonet', 'data=phbreast'])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.input_size = [300, 250]
            cfg.data.batch_size = 4
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope='function')
def cfg_physbonet(cfg_physbonet_global, tmp_path) -> DictConfig:
    cfg = cfg_physbonet_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope='package')
def cfg_physenet_global() -> DictConfig:
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='train.yaml',
                      return_hydra_config=True, overrides=['model=physenet', 'data=phbreast'])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.input_size = [300, 250]
            cfg.data.batch_size = 4
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope='function')
def cfg_physenet(cfg_physenet_global, tmp_path) -> DictConfig:
    cfg = cfg_physenet_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope='package')
def cfg_bcresnet_global() -> DictConfig:
    """Fixture for providing a global configuration for Breast Cancer Classification with ResNet.

    Returns:
        DictConfig: The global configuration for Breast Cancer Classification with ResNet.
    """
    with initialize(version_base='1.3', config_path='../configs'):
        cfg = compose(config_name='train.yaml',
                      return_hydra_config=True, overrides=['model=bcc', 'data=bcc'])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                pyrootutils.find_root(indicator='.project-root'))
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.data.input_size = [256, 256]
            cfg.data.num_workers = 0
            cfg.data.batch_size = 4
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope='function')
def cfg_bcresnet18(cfg_bcresnet_global, tmp_path) -> DictConfig:
    """Fixture for providing a specific configuration for Breast Cancer Classification with ResNet.

    Args:
        cfg_bcresnet_global (DictConfig): Global configuration for Breast Cancer Classification with ResNet.
        tmp_path (str): Temporary path for data.

    Returns:
        DictConfig: The configuration for Breast Cancer Classification with ResNet.
    """
    cfg = cfg_bcresnet_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.model.net._target_ = 'src.models.components.layers.basic.ResNet18'
    yield cfg

    GlobalHydra.instance().clear()
