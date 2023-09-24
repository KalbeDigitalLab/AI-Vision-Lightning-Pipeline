from typing import Optional, Tuple

import hub
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class VinDrBodyPartXRDataset(Dataset):
    """VinDr Body Part XR Parser.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, transform: Optional[torchvision.transforms.Compose] = None, ds=None, stage: str = 'train'):
        if stage == 'train':
            self.ds = hub.load(ds + 'train', read_only=True)
        elif stage == 'val':
            self.ds = hub.load(ds + 'val', read_only=True)
        elif stage == 'test':
            self.ds = hub.load(ds + 'test', read_only=True)
        else:
            print(
                'Invalid Stage Input. There is no dataset with a corresponding dataset on ActiveLoop')
        self.transform = transform

    def __len__(self) -> int:
        """Get number of images."""
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.ds.images[idx].numpy()
        label = self.ds.labels[idx].numpy(fetch_chunks=True).astype(np.int32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # If no transformations, convert to C,H,W
        # Normalize to [0,1]
        if isinstance(image, np.ndarray):
            # image = np.transpose(image, (1, 2, 0))
            # image = torch.from_numpy()
            image /= 255.0

        return image, torch.tensor(label, dtype=torch.int64)
