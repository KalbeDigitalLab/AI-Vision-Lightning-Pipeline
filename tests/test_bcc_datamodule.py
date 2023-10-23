import torch

from src.data.bcc_datamodule import BreastCancerDataModule
from tests.bc_dummy_dataset import (
    deeplake_dummy_dataset_dir,
    deeplake_dummy_test_dataset_dir,
    deeplake_dummy_train_dataset_dir,
    deeplake_dummy_val_dataset_dir,
)


def test_bc_classification_deeplake_lit_datamodule(deeplake_dummy_train_dataset_dir, deeplake_dummy_val_dataset_dir, deeplake_dummy_test_dataset_dir):
    """Test the BreastCancerDataModule with Deeplake for Breast Cancer classification.

    This test ensures that the `BreastCancerDataModule` can be set up with the provided Deeplake dataset paths and that the data can be loaded correctly.

    Parameters:
        deeplake_dummy_train_dataset_dir (str): The path to the dummy Breast Cancer train dataset.
        deeplake_dummy_val_dataset_dir (str): The path to the dummy Breast Cancer validation dataset.
        deeplake_dummy_test_dataset_dir (str): The path to the dummy Breast Cancer test dataset.

    This test checks the following:
    - The data module is properly set up with the provided dataset paths.
    - The data is loaded successfully for training, validation, and testing.
    - The data loaders return non-empty batches.
    - The data types and shapes of the loaded data are as expected.

    Args:
        deeplake_dummy_train_dataset_dir (str): The path to the dummy Breast Cancer train dataset directory.
        deeplake_dummy_val_dataset_dir (str): The path to the dummy Breast Cancer validation dataset directory.
        deeplake_dummy_test_dataset_dir (str): The path to the dummy Breast Cancer test dataset directory.
    """
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
