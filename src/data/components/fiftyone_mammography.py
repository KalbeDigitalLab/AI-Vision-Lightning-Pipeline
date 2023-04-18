import re
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from src.data.components.fiftyone_parser import FiftyOneDatasetParser


class FiftyOneVinDrMammography(FiftyOneDatasetParser):
    # TODO: Docstring

    def __init__(self, num_views: int, path: str, stage: str = 'train', transform: Optional[Any] = None):
        super().__init__(path, stage, transform)

        self._num_views = num_views

        self._group_ids = []
        for sample in self._dataset:
            self._group_ids.append(sample.view_side)

    @property
    def num_views(self):
        # TODO: Docstring
        return self._num_views

    def __getitem__(self, idx: int) -> Any:
        # TODO: Docstring

        # Get group and sort by left-right, cc-mlo
        group = self._dataset.get_group(self._group_ids[idx].id)
        sorted_keys = list(group.keys())
        sorted_keys.sort()
        group = {i: group[i] for i in sorted_keys}
        images_stack = []

        study_ids = []
        breast_birads = []
        breast_density = []

        for g_data in group.values():
            filepath = g_data.filepath
            breast_density.append(g_data.breast_density.label)
            breast_birads.append(g_data.breast_birads.label)
            study_ids.append(str(g_data.study_id))
            image = Image.open(filepath).convert('L')
            images_stack.append(np.asarray(image))

        if len(images_stack) != self._num_views:
            raise ValueError(f'Incorrect number of provided images. \
                             Expected {self._num_views}, got {len(images_stack)}')
        images_stack = np.stack(images_stack, axis=-1)

        breast_birads = [int(re.search(r'\d+', level).group()) for level in breast_birads]
        breast_birads = [0 if level < 3 else 1 for level in breast_birads]

        breast_density = [level.split(' ')[1] for level in breast_density]
        breast_density = [['A', 'B', 'C', 'D'].index(level) for level in breast_density]

        if self._transform is not None:
            images_stack = self._transform(images_stack)

        if not isinstance(images_stack, torch.Tensor):
            images_stack = torch.from_numpy(images_stack).to(torch.float32)
            images_stack /= 255

        return images_stack, breast_birads, breast_density, study_ids
