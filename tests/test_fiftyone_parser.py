from typing import Any

import fiftyone
import pytest

from src.data.components.fiftyone_parser import FiftyOneDatasetParser
from tests.helpers.run_if import RunIf


class QuickstartDataset(FiftyOneDatasetParser):
    def __getitem__(self, idx: int) -> Any:
        sample = self._dataset[self._sample_ids[idx]]
        return sample['filepath']


@RunIf(mongod=True)
@pytest.fixture(scope='session', autouse=True)
def quickstart_dataset_path(tmp_path_factory):
    """Download quickstart dataset."""
    temp_dir = tmp_path_factory.mktemp('temp_quickstart_dataset')
    _, dataset_dir = fiftyone.zoo.download_zoo_dataset(
        'quickstart', dataset_dir=str(temp_dir))
    return dataset_dir


@RunIf(mongod=True)
@pytest.fixture(scope='session', autouse=True)
def torch_mnist_dataset_path(tmp_path_factory):
    """Download quickstart dataset."""
    temp_dir = tmp_path_factory.mktemp('temp_quickstart_dataset')
    _, dataset_dir = fiftyone.zoo.download_zoo_dataset(
        'mnist', dataset_dir=str(temp_dir))
    return dataset_dir


def test_valid_dataset(quickstart_dataset_path):
    dataset = QuickstartDataset(path=quickstart_dataset_path)

    assert len(dataset) == 200
    assert isinstance(dataset[0], str)


def test_invalid_empty_dataset(quickstart_dataset_path):
    dataset = fiftyone.Dataset.from_dir(
        dataset_dir=quickstart_dataset_path, dataset_type=fiftyone.types.FiftyOneDataset)
    dataset.clear()
    dataset.save()
    dataset.export(
        export_dir=quickstart_dataset_path,
        dataset_type=fiftyone.types.FiftyOneDataset,
    )
    del dataset

    with pytest.raises(Exception) as exc_info:
        _ = QuickstartDataset(path=quickstart_dataset_path)

    assert str(exc_info.value) == 'The dataset has 0 samples.'


def test_invalid_type_dataset(torch_mnist_dataset_path):
    with pytest.raises(Exception) as exec_info:
        _ = QuickstartDataset(path=torch_mnist_dataset_path)
