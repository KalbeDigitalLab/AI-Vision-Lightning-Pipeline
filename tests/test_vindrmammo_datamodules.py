from pathlib import Path

import pytest
import torch

from src.data.vindrmammo_datamodule import VinDrLitDatamodule
from tests.dummy_dataset import (
    vindr_2views_mammography_single_dataset_dir,
    vindr_4views_mammography_multi_dataset_dir,
    vindr_4views_mammography_single_dataset_dir,
)


def test_catch_invalid_input(vindr_2views_mammography_single_dataset_dir):
    with pytest.raises(Exception) as e_info:
        _ = VinDrLitDatamodule(
            data_dir=vindr_2views_mammography_single_dataset_dir,
            output_type='singular',
            batch_size=2,
            num_views=2,
            num_classes=1)
        assert e_info == 'Unsupported output type singular. Only [single. multiple] are supported'


def test_vindr_2views_singlehead_datamodule(vindr_2views_mammography_single_dataset_dir):
    dm = VinDrLitDatamodule(
        data_dir=vindr_2views_mammography_single_dataset_dir,
        output_type='single',
        batch_size=2,
        num_views=2,
        num_classes=1)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, breast_classes, _, _ = batch
    assert breast_classes.shape == torch.Size([2, 1])
    assert images_stack.shape[:2] == torch.Size([2, 2])
    assert images_stack.dtype == torch.float32
    assert breast_classes.dtype == torch.int64


def test_vindr_4views_singlehead_datamodule(vindr_4views_mammography_single_dataset_dir):
    dm = VinDrLitDatamodule(
        data_dir=vindr_4views_mammography_single_dataset_dir,
        output_type='single',
        batch_size=2,
        num_views=4,
        num_classes=1)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, breast_classes, _, _ = batch
    assert breast_classes.shape == torch.Size([2, 1])
    assert images_stack.shape[:2] == torch.Size([2, 4])
    assert images_stack.dtype == torch.float32
    assert breast_classes.dtype == torch.int64


def test_vindr_4views_multihead_datamodule(vindr_4views_mammography_multi_dataset_dir):
    dm = VinDrLitDatamodule(
        data_dir=vindr_4views_mammography_multi_dataset_dir,
        output_type='multiple',
        batch_size=2,
        num_views=4,
        num_classes=1)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, breast_classes, _, _ = batch
    assert breast_classes.shape == torch.Size([2, 2])
    assert images_stack.shape[:2] == torch.Size([2, 4])
    assert images_stack.dtype == torch.float32
    assert breast_classes.dtype == torch.int64
