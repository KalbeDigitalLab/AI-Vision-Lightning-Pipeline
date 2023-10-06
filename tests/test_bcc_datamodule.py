import pytest 
import torch 

from src.data.bcc_datamodule import DeepLakeLitDataModule
from tests.bc_dummy_dataset import (dummy_dataset,
                                    dummy_train_dataset, 
                                    dummy_val_dataset, 
                                    dummy_test_dataset
                                      )

def test_bc_classification_deeplake_lit_datamodule(dummy_train_dataset, dummy_val_dataset, dummy_test_dataset):
    dm = DeepLakeLitDataModule(
        train_dir=dummy_train_dataset,
        val_dir=dummy_val_dataset,
        test_dir=dummy_test_dataset
    )
    assert not dm.data_train and not dm.data_val and not dm.data_test
    
    dm.setup()
    
    assert dm.data_train  and dm.data_val and dm.data_test
    
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    assert train_dataloader and val_dataloader and test_dataloader
    assert len(dm.data_train) + len(dm.data_val) + len(dm.data_test)== 935
    
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