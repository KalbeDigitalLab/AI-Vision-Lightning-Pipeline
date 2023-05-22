import os
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import Dataset


class PHBreastZIPDataset(Dataset):
    """PHBreast Dataset Parser.

    Dataset loader from PHBReast dataset.
    # image [4, 600, 500] label: list [0, 1] one-hot

    Parameters
    ----------
    num_views : int
        Number of views for the dataset, either 2 or 4.
    path : str
        Path to dataset.
    stage : str, optional
        Stage to apply augmentation, by default None
    transform : Any, optional
        Transformation function, by default None

    Raises
    ------
    ValueError
        If the given path is not a valid directory/file.
    RuntimeError
        If the final path to file is not valid.
    """

    def __init__(self, num_views: int, path: str, stage: Optional[str] = None, transform: Optional[Any] = None):
        file_path: str = ''
        if os.path.isdir(path):
            files = sorted(os.listdir(path))
            if stage is None:
                file_path = os.path.join(path, files[0])
            else:
                for p in files:
                    if stage in p:
                        file_path = os.path.join(path, p)

        else:
            file_path = path

        if file_path == '':
            raise RuntimeError('Invalid path to file')

        self._num_views = num_views
        self._data = torch.load(file_path)
        self._transform = transform

    @property
    def num_views(self) -> int:
        """Get dataset number of views."""
        return self._num_views

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample by index."""
        images, label = self._data[idx]
        if isinstance(label, int):
            label = [label]
        if len(images) != self._num_views:
            raise ValueError(f'Incorrect number of provided images. \
                             Expected {self._num_views}, got {len(images)}')

        if self._transform:
            images = self._transform(images)

        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).to(torch.float32)

        return images, torch.tensor(label, dtype=torch.int64)
