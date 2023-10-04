import torch
import torchvision.transforms as transforms
import deeplake
from src.data.components.bcc_parser import DeepLakeDataset
import os
from tests.bc_dummy_dataset import (dummy_dataset,
                                    dummy_train_dataset, 
                                    dummy_val_dataset, 
                                    dummy_test_dataset
                                      )

def train_dataset(dummy_train_dataset):
    dataset = DeepLakeDataset(ds=dummy_train_dataset, transform=transforms.Compose([
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

def val_dataset(dummy_val_dataset):
    dataset = DeepLakeDataset(ds=dummy_val_dataset, transform=transforms.Compose([
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

def test_dataset(dummy_test_dataset):
    dataset = DeepLakeDataset(ds=dummy_test_dataset, transform=transforms.Compose([
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
    