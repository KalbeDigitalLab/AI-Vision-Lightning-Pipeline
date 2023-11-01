import pytest
import deeplake
import torch
import torchvision.transforms as transforms
from src.data.components.ham1000_parser import ham1000dataset
from tests.ham_dummy_dataset import (
    ham_dummy_train_dataset_dir,
    ham_dummy_val_dataset_dir,
    ham_dummy_test_dataset_dir,
)

def test_dummy_ham_train_dataset(ham_dummy_train_dataset_dir):
    dataset = ham1000dataset(ham_dummy_train_dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    image, label = dataset[10]

    assert len(dataset) == 547
    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)

def test_dummy_ham_val_dataset(ham_dummy_val_dataset_dir):
    dataset = ham1000dataset(ham_dummy_val_dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    image, label = dataset[10]

    assert len(dataset) == 233
    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)

def test_dummy_ham_test_dataset(ham_dummy_test_dataset_dir):
    dataset = ham1000dataset(ham_dummy_test_dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    assert len(dataset) == 155
    image, label = dataset[10]
    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)

def test_invalid_data_dir_type():
    with pytest.raises(TypeError):
        dataset = ham1000dataset(123, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1])
        ]))
        
def test_invalid_data_dir_type():
    
    with pytest.raises(TypeError):
        dataset = ham1000dataset(data_dir=123, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1])
        ]))

def test_deeplake_data_type(ham_dummy_train_dataset_dir):
    data_ham = deeplake.load(ham_dummy_train_dataset_dir)
    dataset = ham1000dataset(dh=data_ham, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]))

    assert len(dataset) == 547
    image, label = dataset[10]
    assert image.shape == (1, 224, 224)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)