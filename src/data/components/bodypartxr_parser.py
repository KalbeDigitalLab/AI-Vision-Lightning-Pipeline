from typing import Optional, Tuple

import deeplake
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class VinDrBodyPartXRDataset(Dataset):
    """VinDr Body Part XR Parser. This dataset from VinBigData contains x-ray imaging from 5
    different class in DICOM format.

    The dataset is stored using ActiveLoop for easy integration.

    Dataset source: https://vindr.ai/datasets/bodypartxr
    ActiveLoop: https://app.activeloop.ai/

    Parameters
    ----------
    dataset_dir : str
        Activeloop dataset directory
    transform : Optional[torchvision.transforms.Compose], optional
        Data augmentation pipeline, by default None
    """

    def __init__(self, transform: Optional[torchvision.transforms.Compose] = None, dataset_dir: Optional[str] = None):
        self.ds = deeplake.load(dataset_dir)
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

        # If no transformations
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(torch.float32)

        return image, torch.tensor(label, dtype=torch.int64)
