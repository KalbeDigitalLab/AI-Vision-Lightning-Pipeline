import deeplake
import pytest
import torch
import torchvision.transforms as transforms

from src.data.components.deeplake_parser import DeepLakeDataset
from tests.bc_dummy_dataset import (
    deeplake_dummy_dataset_dir,
    deeplake_dummy_test_dataset_dir,
    deeplake_dummy_train_dataset_dir,
    deeplake_dummy_val_dataset_dir,
)


def test_dummy_bcc_train_dataset(deeplake_dummy_train_dataset_dir):
    """Test the DeepLakeDataset class with the dummy Breast Cancer train dataset.

    This test ensures that the DeepLakeDataset class can load and transform data correctly from a dummy Breast Cancer train dataset.

    Parameters:
        deeplake_dummy_train_dataset_dir (str): The path to the dummy Breast Cancer train dataset.

    This test checks the following:
    - The dataset length is as expected (547 samples).
    - The shape and data type of the loaded images and labels are correct.

    Args:
        deeplake_dummy_train_dataset_dir (str): The path to the dummy Breast Cancer train dataset directory.
    """
    dataset = DeepLakeDataset(data_dir=deeplake_dummy_train_dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    image, label = dataset[10]

    print(len(dataset))

    assert len(dataset) == 547
    assert image.shape == (1, 256, 256)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_dummy_bcc_val_dataset(deeplake_dummy_val_dataset_dir):
    """Test the DeepLakeDataset class with the dummy Breast Cancer validation dataset.

    This test ensures that the DeepLakeDataset class can load and transform data correctly from a dummy Breast Cancer validation dataset.

    Parameters:
        deeplake_dummy_val_dataset_dir (str): The path to the dummy Breast Cancer validation dataset.

    This test checks the following:
    - The dataset length is as expected (233 samples).
    - The shape and data type of the loaded images and labels are correct.

    Args:
        deeplake_dummy_val_dataset_dir (str): The path to the dummy Breast Cancer validation dataset directory.
    """
    dataset = DeepLakeDataset(data_dir=deeplake_dummy_val_dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    image, label = dataset[10]

    print(len(dataset))

    assert len(dataset) == 233
    assert image.shape == (1, 256, 256)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_dummy_bcc_test_dataset(deeplake_dummy_test_dataset_dir):
    """Test the DeepLakeDataset class with the dummy Breast Cancer test dataset.

    This test ensures that the DeepLakeDataset class can load and transform data correctly from a dummy Breast Cancer test dataset.

    Parameters:
        deeplake_dummy_test_dataset_dir (str): The path to the dummy Breast Cancer test dataset.

    This test checks the following:
    - The dataset length is as expected (155 samples).
    - The shape and data type of the loaded images and labels are correct.

    Args:
        deeplake_dummy_test_dataset_dir (str): The path to the dummy Breast Cancer test dataset directory.
    """
    dataset = DeepLakeDataset(data_dir=deeplake_dummy_test_dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    print(len(dataset))

    image, label = dataset[10]
    assert len(dataset) == 155
    assert image.shape == (1, 256, 256)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_invalid_data_dir_type():
    """Test the DeepLakeDataset class with an invalid data_dir type.

    This test checks that an error is raised when providing an invalid data_dir type to the DeepLakeDataset class.

    This test checks the following:
    - An error is raised when an invalid data_dir (integer) is provided.

    Raises:
        TypeError: If the data_dir parameter is not a valid path or Deeplake dataset object.
    """
    with pytest.raises(TypeError):
        dataset = DeepLakeDataset(data_dir=123, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1])
        ]))


def test_deeplake_data_type(deeplake_dummy_train_dataset_dir):
    """Test the DeepLakeDataset class with a Deeplake data object.

    This test ensures that the DeepLakeDataset class can load and transform data correctly from a Deeplake dataset object.

    Parameters:
        deeplake_dummy_train_dataset_dir (str): The path to the dummy Breast Cancer train dataset.

    This test checks the following:
    - The dataset length is as expected (547 samples).
    - The shape and data type of the loaded images and labels are correct.

    Args:
        deeplake_dummy_train_dataset_dir (str): The path to the dummy Breast Cancer train dataset directory.
    """
    data_deeplake = deeplake.load(deeplake_dummy_train_dataset_dir)
    dataset = DeepLakeDataset(data_dir=data_deeplake, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    print(len(dataset))

    image, label = dataset[10]
    assert len(dataset) == 547
    assert image.shape == (1, 256, 256)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
