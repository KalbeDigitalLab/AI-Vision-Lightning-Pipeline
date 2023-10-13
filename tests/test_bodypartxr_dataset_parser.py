import pytest
import torch
from torchvision import transforms

from src.data.components.bodypartxr_parser import VinDrBodyPartXRDataset
from tests.dummy_bodypartxr_dataset import dummy_test_dataset, dummy_train_dataset, dummy_val_dataset


def test_train_dummy_dataset(dummy_train_dataset):
    dataset = VinDrBodyPartXRDataset(dataset_dir=dummy_train_dataset, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])]))

    assert len(dataset) == 50

    image, label = dataset[1]

    assert image.shape == (1, 384, 384)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_val_dummy_dataset(dummy_val_dataset):
    dataset = VinDrBodyPartXRDataset(dataset_dir=dummy_val_dataset, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])]))

    assert len(dataset) == 30

    image, label = dataset[1]

    assert image.shape == (1, 384, 384)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_testing_dummy_dataset(dummy_test_dataset):
    dataset = VinDrBodyPartXRDataset(dataset_dir=dummy_test_dataset, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])]))

    assert len(dataset) == 25

    image, label = dataset[1]

    assert image.shape == (1, 384, 384)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
