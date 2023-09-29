import pytest
import torch

from src.data.bodypartxr_datamodule import VinDrBodyPartXRDatamodule
from tests.dummy_bodypartxr_dataset import dummy_test_dataset, dummy_train_dataset, dummy_val_dataset


def test_bodypartxr_datamodule(dummy_train_dataset, dummy_val_dataset, dummy_test_dataset):
    dm = VinDrBodyPartXRDatamodule(
        train_dir=dummy_train_dataset, val_dir=dummy_val_dataset, test_dir=dummy_test_dataset, batch_size=4)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()
    assert train_dataloader and val_dataloader and test_dataloader

    # train = 50, val = 30, test = 25
    assert len(dm.data_train) + len(dm.data_val) + len(dm.data_test) == 105

    batch = next(iter(train_dataloader))
    x, y = batch
    assert len(x) == 4
    assert len(y) == 4
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
