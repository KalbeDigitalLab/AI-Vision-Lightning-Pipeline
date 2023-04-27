import re
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.components.fiftyone_parser import FiftyOneDatasetParser


class FiftyOneVinDrMammography(FiftyOneDatasetParser):
    """VinDrMammography Dataset Parser.

    The dataset is stored using FiftyOneDataset format for easy visualization and integration.
    Dataset source: https://physionet.org/content/vindr-mammo/1.0.0/
    Dataset format: FiftyOne Dataset with grouping by view_side.
                    A single group must have left_cc, left_mlo, right_cc, right_mlo.
    Available annotations:
        - Breast Density [A, B, C, D] -> [0, 1, 2, 3]
        - Breast Birads [1, 2, 3, 4, 5] -> [0, 1] (Normal/bening, Malignant)
        - Study ids

    Parameters
    ----------
    num_views : int
        Number of views for the dataset, either 2 or 4.
    path : str
        Path to dataset.
    stage : str, optional
        Stage to slice dataset and apply augmentation, by default None.
    transform : Optional[Any], optional
        Transformation callback, by default None.
    """

    def __init__(self, num_views: int, path: str, stage: Optional[str] = None, transform: Optional[Any] = None):
        super().__init__(path, stage, transform)

        if self._stage is not None:
            self._dataset = self._dataset.match_tags(self._stage)

        self._num_views = num_views
        self._group_ids = []
        for sample in self._dataset:
            self._group_ids.append(sample.view_side)

        # * NOTE: fiftyone.Dataset is not pickle-able.
        # * This will cause problem when using DDP
        group_samples = [self._dataset.get_group(ids.id) for ids in self._group_ids]
        self.group_samples = []
        for sample in group_samples:
            g_samples = {}
            for key, val in sample.items():
                g_samples[key] = val.to_dict()
            self.group_samples.append(g_samples)
        self._dataset = None

    @property
    def num_views(self) -> int:
        """Get dataset number of views."""
        return self._num_views

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        return len(self.group_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List, List, List]:
        """Get samples from dataset.

        Given an index, get the samples from group index.

        Parameters
        ----------
        idx : int
            Requested group index

        Returns
        -------
        Tuple
            Tuple of images, breast_birads, breast_density, and study_ids

        Raises
        ------
        ValueError
            _description_
        """

        # Get group and sort by left-right, cc-mlo
        group = self.group_samples[idx]  # noqa: F821
        sorted_keys = list(group.keys())
        sorted_keys.sort()
        group = {i: group[i] for i in sorted_keys}
        images_stack = []

        study_ids = []
        breast_birads = []
        breast_density = []

        for g_data in group.values():
            filepath = g_data['filepath']
            breast_density.append(g_data['breast_density']['label'])
            breast_birads.append(g_data['breast_birads']['label'])
            study_ids.append(str(g_data['study_id']))
            image = Image.open(filepath).convert('L')
            images_stack.append(np.asarray(image))

        if len(images_stack) != self._num_views:
            raise ValueError(f'Incorrect number of provided images. \
                             Expected {self._num_views}, got {len(images_stack)}')
        images_stack = np.stack(images_stack, axis=-1)

        # Forcing to normal/bening or malignant
        breast_birads = [int(re.search(r'\d+', level).group()) for level in breast_birads]
        breast_birads = list({0 if level < 3 else 1 for level in breast_birads})
        if len(breast_birads) != 1:
            raise RuntimeError(
                f'The breast birads is not unique for the given group. This group has {breast_birads}')

        breast_density = [level.split(' ')[1] for level in breast_density]
        breast_density = list({['A', 'B', 'C', 'D'].index(level) for level in breast_density})
        if len(breast_density) != 1:
            raise RuntimeError(
                f'The breast density is not unique for the given group. This group has {breast_density}')

        if self._transform is not None:
            images_stack = self._transform(images_stack)

        if not isinstance(images_stack, torch.Tensor):
            images_stack = torch.from_numpy(images_stack).to(torch.float32)
            images_stack /= 255

        return images_stack, torch.tensor(breast_birads, dtype=torch.int64), torch.tensor(breast_density, dtype=torch.int64), study_ids

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        images_stack, breast_birads, breast_density, study_ids = zip(*batch)
        return torch.stack(images_stack, 0), torch.cat(breast_birads, 0), torch.cat(breast_density, 0), study_ids
