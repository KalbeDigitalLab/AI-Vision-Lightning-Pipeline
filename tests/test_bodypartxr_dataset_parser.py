import pytest
import torch
from torchvision import transforms

from src.data.components.bodypartxr_parser import VinDrBodyPartXRDataset

DATA_PATH = 'hub://jasonadrianzoom/bodyclassification'


def test_train_dataset():
    dataset = VinDrBodyPartXRDataset(ds=DATA_PATH, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]), stage='train')

    assert len(dataset) == 9819

    image, label = dataset[10]

    assert image.shape == (1, 384, 384)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_validation_dataset():
    dataset = VinDrBodyPartXRDataset(ds=DATA_PATH, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]), stage='val')

    assert len(dataset) == 2108

    image, label = dataset[12]

    assert image.shape == (1, 250, 250)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_testing_dataset():
    dataset = VinDrBodyPartXRDataset(ds=DATA_PATH, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ]), stage='test')

    assert len(dataset) == 2103

    image, label = dataset[1]

    assert image.shape == (1, 250, 250)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
