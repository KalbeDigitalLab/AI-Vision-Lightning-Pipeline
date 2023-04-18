import pytest
import torch
from torchvision import transforms

from src.data.components.fiftyone_mammography import FiftyOneVinDrMammography
from tests.dummy_dataset import empty_dataset_dir, vindr_mammography_dataset_dir


def test_load_dataset(vindr_mammography_dataset_dir):
    fo_dataset = FiftyOneVinDrMammography(num_views=4, path=vindr_mammography_dataset_dir)
    assert len(fo_dataset) == 40

    images_stack, breast_birads, breast_density, study_ids = fo_dataset[0]
    assert images_stack.shape == (384, 384, 4)
    assert isinstance(images_stack, torch.Tensor)
    assert len(breast_birads) == 4
    assert len(breast_density) == 4
    assert len(study_ids) == 4


def test_dataset_transforms(vindr_mammography_dataset_dir):
    augmentation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((600, 500)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-25, 25)),
        transforms.Normalize((0, 0, 0, 0), (1, 1, 1, 1)),
    ])

    fo_dataset = FiftyOneVinDrMammography(
        num_views=4, path=vindr_mammography_dataset_dir, transform=augmentation)

    images_stack, breast_birads, breast_density, study_ids = fo_dataset[0]
    assert images_stack.shape == (4, 600, 500)
    assert len(breast_birads) == 4
    assert len(breast_density) == 4
    assert len(study_ids) == 4


def test_dataset_empty(empty_dataset_dir):
    with pytest.raises(Exception) as e_info:
        _ = FiftyOneVinDrMammography(num_views=4, path=empty_dataset_dir)
        assert e_info == 'The dataset has 0 samples.'
