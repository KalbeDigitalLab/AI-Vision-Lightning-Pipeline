import pytest
import torch
from torchvision import transforms

from src.data.components.phbreast_dataset_parser import PHBreastZIPDataset
from tests.dummy_dataset import (
    empty_dataset_dir,
    inbreast_2views_multi_dataset_dir,
    inbreast_2views_multi_dataset_path,
    inbreast_2views_patches_dataset_path,
    inbreast_2views_single_dataset_path,
)


def test_load_multihead_2views_dataset_path(inbreast_2views_multi_dataset_path):
    dataset = PHBreastZIPDataset(num_views=2, path=inbreast_2views_multi_dataset_path)

    images_stack, labels = dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert labels.shape == torch.Size([2])
    assert dataset.num_views == 2
    assert len(dataset) == 40


def test_load_singlehead_2views_dataset_path(inbreast_2views_single_dataset_path):
    dataset = PHBreastZIPDataset(num_views=2, path=inbreast_2views_single_dataset_path)

    images_stack, labels = dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert labels.shape == torch.Size([1])
    assert dataset.num_views == 2
    assert len(dataset) == 40


def test_load_multihead_2views_dataset_dir(inbreast_2views_multi_dataset_dir):
    dataset = PHBreastZIPDataset(num_views=2, path=inbreast_2views_multi_dataset_dir)

    images_stack, labels = dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert labels.shape == torch.Size([2])
    assert dataset.num_views == 2
    assert len(dataset) == 40

    dataset = PHBreastZIPDataset(num_views=2, path=inbreast_2views_multi_dataset_dir, stage='train')

    images_stack, labels = dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert labels.shape == torch.Size([2])
    assert dataset.num_views == 2
    assert len(dataset) == 40

    dataset = PHBreastZIPDataset(num_views=2, path=inbreast_2views_multi_dataset_dir, stage='valid')

    images_stack, labels = dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert labels.shape == torch.Size([2])
    assert dataset.num_views == 2
    assert len(dataset) == 96


def test_load_singlehead_patches_dataset(inbreast_2views_patches_dataset_path):
    dataset = PHBreastZIPDataset(num_views=2, path=inbreast_2views_patches_dataset_path)

    images_stack, labels = dataset[0]
    print(labels)
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert labels.shape == torch.Size([1])
    assert dataset.num_views == 2
    assert len(dataset) == 40


def test_multihead_2views_dataset_transforms(inbreast_2views_multi_dataset_dir):
    augmentation = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-25, 25)),
    ])

    dataset = PHBreastZIPDataset(
        num_views=2, path=inbreast_2views_multi_dataset_dir, transform=augmentation)

    images_stack, labels = dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert labels.shape == torch.Size([2])


def test_invalid_dataset_dir():
    with pytest.raises(Exception) as e_info:
        _ = PHBreastZIPDataset(num_views=2, path='')
