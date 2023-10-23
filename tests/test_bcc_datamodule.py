import torch

from src.data.bcc_datamodule import BreastCancerDataModule
from tests.bc_dummy_dataset import (
    deeplake_dummy_dataset_dir,
    deeplake_dummy_test_dataset_dir,
    deeplake_dummy_train_dataset_dir,
    deeplake_dummy_val_dataset_dir,
)


def test_bc_classification_deeplake_lit_datamodule(deeplake_dummy_train_dataset_dir, deeplake_dummy_val_dataset_dir, deeplake_dummy_test_dataset_dir):
    dm = BreastCancerDataModule(
        train_dir=deeplake_dummy_train_dataset_dir,
        val_dir=deeplake_dummy_val_dataset_dir,
        test_dir=deeplake_dummy_test_dataset_dir
    )
    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()

    assert dm.data_train and dm.data_val and dm.data_test

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    assert train_dataloader and val_dataloader and test_dataloader
    assert len(dm.data_train) + len(dm.data_val) + len(dm.data_test) == 935

    batch = next(iter(train_dataloader))
    x, y = batch
    assert len(x) == 64
    assert len(y) == 64
    print('len(x):', len(x))
    print('len(y):', len(y))

    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    print('type of x: ', x.dtype)
    print('type of y: ', y.dtype)
