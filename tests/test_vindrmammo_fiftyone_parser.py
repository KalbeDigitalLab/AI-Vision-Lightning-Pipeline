import pytest
import torch
from torchvision import transforms

from src.data.components.fiftyone_mammography import VinDrMammographyDataset
from tests.dummy_dataset import (
    empty_dataset_dir,
    vindr_2views_mammography_single_dataset_dir,
    vindr_4views_mammography_multi_dataset_dir,
    vindr_4views_mammography_single_dataset_dir,
)


def test_load_singlehead_2views_dataset(vindr_2views_mammography_single_dataset_dir):
    fo_dataset = VinDrMammographyDataset(num_views=2, output_type='single',
                                         path=vindr_2views_mammography_single_dataset_dir)

    images_stack, breast_birads, breast_density, study_ids = fo_dataset[0]
    assert images_stack.shape == (2, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert isinstance(breast_birads, torch.Tensor)
    assert breast_birads.shape == torch.Size([1])
    assert isinstance(breast_density, torch.Tensor)
    assert breast_density.shape == torch.Size([1])
    assert len(study_ids) == 2


def test_load_singlehead_4views_dataset(vindr_4views_mammography_single_dataset_dir):
    fo_dataset = VinDrMammographyDataset(num_views=4, output_type='single',
                                         path=vindr_4views_mammography_single_dataset_dir)

    images_stack, breast_birads, breast_density, study_ids = fo_dataset[0]
    assert images_stack.shape == (4, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert isinstance(breast_birads, torch.Tensor)
    assert breast_birads.shape == torch.Size([1])
    assert isinstance(breast_density, torch.Tensor)
    assert breast_density.shape == torch.Size([1])
    assert len(study_ids) == 4


def test_singlehead_4views_dataset_transforms(vindr_4views_mammography_single_dataset_dir):
    augmentation = transforms.Compose([
        transforms.Resize((600, 500)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-25, 25)),
    ])

    fo_dataset = VinDrMammographyDataset(
        num_views=4, output_type='single', path=vindr_4views_mammography_single_dataset_dir, transform=augmentation)

    images_stack, breast_birads, breast_density, study_ids = fo_dataset[0]
    assert images_stack.shape == (4, 600, 500)
    assert breast_birads.shape == torch.Size([1])
    assert breast_density.shape == torch.Size([1])
    assert len(study_ids) == 4


def test_dataset_empty(empty_dataset_dir):
    with pytest.raises(Exception) as e_info:
        _ = VinDrMammographyDataset(num_views=4, output_type='single', path=empty_dataset_dir)
        assert e_info == 'The dataset has 0 samples.'


def test_load_multihead_4views_dataset(vindr_4views_mammography_multi_dataset_dir):
    fo_dataset = VinDrMammographyDataset(num_views=4, output_type='multiple',
                                         path=vindr_4views_mammography_multi_dataset_dir)

    images_stack, breast_birads, breast_density, study_ids = fo_dataset[0]
    assert images_stack.shape == (4, 384, 384)
    assert isinstance(images_stack, torch.Tensor)
    assert isinstance(breast_birads, torch.Tensor)
    assert breast_birads.shape == torch.Size([2])
    assert isinstance(breast_density, torch.Tensor)
    assert breast_density.shape == torch.Size([1])
    assert len(study_ids) == 4
