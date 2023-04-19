import json
import os
from typing import Any, Optional

import fiftyone
from torch.utils.data import Dataset


class FiftyOneDatasetParser(Dataset):
    """FiftyoneDataset format parser.

    Parse FiftyoneDataset format to torch Dataset.
    Reference: https://docs.voxel51.com/api/fiftyone.types.html#fiftyone.types.FiftyOneDataset

    Parameters
    ----------
    path : str
        Path to dataset.
    stage : str, optional
        Stage to apply augmentation, by default 'train'
    transform : Any, optional
        Transformation function, by default None

    Raises
    ------
    ValueError
        If the dataset has 0 samples.
    """

    def __init__(self, path: str, stage: str = 'train', transform: Optional[Any] = None):
        self._dataset = fiftyone.Dataset.from_dir(
            dataset_dir=path,
            dataset_type=fiftyone.types.FiftyOneDataset,
        )

        self._dataset = self._dataset.match_tags(stage)

        if len(self._dataset) == 0:
            raise ValueError('The dataset has 0 samples.')

        with open(os.path.join(path, 'samples.json')) as f:
            metadata = json.load(f)
            self._sample_ids = {idx: os.path.join(path, sample['filepath'])
                                for idx, sample in enumerate(metadata['samples'])}

        self._stage = stage
        self._transform = transform

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        """Get sample by index."""
        raise NotImplementedError
