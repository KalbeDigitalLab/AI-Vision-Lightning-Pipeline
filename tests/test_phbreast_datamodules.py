from pathlib import Path

import pytest
import torch

from src.data.phbreast_datamodule import PHBReastLitDatamodule
from tests.dummy_dataset import (
    empty_dataset_dir,
    inbreast_2views_multi_dataset_dir,
    inbreast_2views_patches_dataset_path,
    inbreast_2views_single_dataset_path,
)


def test_phbreast_2views_singlehead_datamodule(inbreast_2views_single_dataset_path):
    dm = PHBReastLitDatamodule(
        data_dir=inbreast_2views_single_dataset_path,
        batch_size=2,
        num_views=2,
        num_classes=1)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, labels = batch
    assert len(images_stack) == 2
    assert images_stack.dtype == torch.float32
    assert labels.shape == torch.Size([2, 1])
    assert labels.dtype == torch.int64


def test_phbreast_2views_multihead_datamodule(inbreast_2views_multi_dataset_dir):
    dm = PHBReastLitDatamodule(
        data_dir=inbreast_2views_multi_dataset_dir,
        batch_size=2,
        num_views=2,
        num_classes=1)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, labels = batch
    assert len(images_stack) == 2
    assert images_stack.dtype == torch.float32
    assert labels.shape == torch.Size([2, 2])
    assert labels.dtype == torch.int64


def test_phbreast_patches_singlehead_datamodule(inbreast_2views_patches_dataset_path):
    dm = PHBReastLitDatamodule(
        data_dir=inbreast_2views_patches_dataset_path,
        batch_size=2,
        num_views=2,
        num_classes=1)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, labels = batch
    assert len(images_stack) == 2
    assert images_stack.dtype == torch.float32
    assert labels.shape == torch.Size([2, 1])
    assert labels.dtype == torch.int64
