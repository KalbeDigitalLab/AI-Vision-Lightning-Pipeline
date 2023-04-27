from pathlib import Path

import pytest
import torch

from src.data.phbreast_datamodule import PHBReastLitDatamodule
from tests.dummy_dataset import vindr_2views_mammography_dataset_dir


def test_vindr_2views_datamodule(vindr_2views_mammography_dataset_dir):
    dm = PHBReastLitDatamodule(
        data_dir=vindr_2views_mammography_dataset_dir,
        batch_size=2,
        num_views=2,
        num_classes=2)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup('train')
    dm.setup('test')
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    images_stack, breast_classes, _, _ = batch
    assert len(images_stack) == 2
    assert len(breast_classes) == 2
    assert images_stack.dtype == torch.float32
    assert breast_classes.dtype == torch.int64
