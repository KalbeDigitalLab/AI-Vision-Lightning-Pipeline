import pytest
import torch

from src.data.bodypartxr_datamodule import VinDrBodyPartXRDatamodule
from tests.dummy_dataset import (
    bodypartxr_dummy_test_dataset_dir,
    bodypartxr_dummy_train_dataset_dir,
    bodypartxr_dummy_val_dataset_dir,
)


def test_bodypartxr_datamodule(bodypartxr_dummy_train_dataset_dir, bodypartxr_dummy_val_dataset_dir, bodypartxr_dummy_test_dataset_dir):
    dm = VinDrBodyPartXRDatamodule(
        train_dir=bodypartxr_dummy_train_dataset_dir, val_dir=bodypartxr_dummy_val_dataset_dir, test_dir=bodypartxr_dummy_test_dataset_dir, batch_size=4)

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
