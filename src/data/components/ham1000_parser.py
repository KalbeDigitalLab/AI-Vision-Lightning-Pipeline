from typing import Optional, Tuple
import deeplake
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class ham1000dataset(Dataset):
    """ham1000dataset. A PyTorch Dataset Wrapper for DeepLake Datasets.

    Parameters:
        dh (str or deeplake.Dataset): The DeepLake dataset object or the path to the dataset directory. If `dh` is a string, it should be the path to the DeepLake dataset directory. If `dh` is a deeplake.Dataset object, it represents the dataset directly.
        transform (Optional[torchvision.transforms.Compose], optional): A data augmentation pipeline to apply to the images. Defaults to None.

    Raises:
        TypeError: If the `dh` parameter is neither a valid DeepLake dataset object nor a string.

    Attributes:
        transform (Optional[torchvision.transforms.Compose]): A data augmentation pipeline for transforming the images.

    Methods:
        __len__(): Get the number of samples in the dataset.
        __getitem__(idx): Get a sample for the given index.
    """

    def __init__(self, dh, transform: Optional[torchvision.transforms.Compose] = None):
        self.transform = transform
        if isinstance(dh, str):
            self.dh = deeplake.load(dh)  # path direct
        elif isinstance(dh, deeplake.Dataset):
            self.dh = dh  # deeplake format
        else:
            raise TypeError('Invalid Format')

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dh)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample for the given index.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its corresponding label as PyTorch tensors.
        """
        image = self.dh.images[idx].numpy()    # Convert the deeplake tensor to numpy array
        label = self.dh.labels[idx].numpy(fetch_chunks=True).astype(np.int32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # normalize to [0, 1]
        if isinstance(image, np.ndarray):
            image /= 255.0

        return image, torch.tensor(label, dtype=torch.int64)