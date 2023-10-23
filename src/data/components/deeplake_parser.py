from typing import Optional, Tuple

import deeplake
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class DeepLakeDataset(Dataset):
    """DeepLakeDataset. A PyTorch Dataset Wrapper for DeepLake Datasets.

    This class serves as a PyTorch Dataset wrapper for DeepLake datasets. It allows you to seamlessly integrate DeepLake datasets into your PyTorch-based machine learning pipelines.

    Parameters:
        data_dir (str or deeplake.Dataset): The DeepLake dataset object or the path to the dataset directory. If `data_dir` is a string, it should be the path to the DeepLake dataset directory. If `data_dir` is a deeplake.Dataset object, it represents the dataset directly.
        transform (Optional[torchvision.transforms.Compose], optional): A data augmentation pipeline to apply to the images. Defaults to None.

    Raises:
        TypeError: If the `data_dir` parameter is neither a valid DeepLake dataset object nor a string.

    Attributes:
        transform (Optional[torchvision.transforms.Compose]): A data augmentation pipeline for transforming the images.

    Methods:
        __len__(): Get the number of samples in the dataset.
        __getitem__(idx): Get a sample for the given index.
    """

    def __init__(self, data_dir, transform: Optional[torchvision.transforms.Compose] = None):
        """Initialize a DeepLakeDataset object.

        Parameters:
            data_dir (str or deeplake.Dataset): The DeepLake dataset object or the path to the dataset directory. If `data_dir` is a string, it should be the path to the DeepLake dataset directory. If `data_dir` is a deeplake.Dataset object, it represents the dataset directly.
            transform (Optional[torchvision.transforms.Compose], optional): A data augmentation pipeline to apply to the images. Defaults to None.

        Raises:
            TypeError: If the `data_dir` parameter is neither a valid DeepLake dataset object nor a string.
        """
        self.transform = transform
        if isinstance(data_dir, str):
            self.data_dir = deeplake.load(data_dir)  # path direct
        elif isinstance(data_dir, deeplake.Dataset):
            self.data_dir = data_dir  # deeplake format
        else:
            raise TypeError('Invalid Format')

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data_dir)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample for the given index.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its corresponding label as PyTorch tensors.
        """
        image = self.data_dir.images[idx].numpy()    # Convert the deeplake tensor to numpy array
        label = self.data_dir.labels[idx].numpy(fetch_chunks=True).astype(np.int32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # normalize to [0, 1]
        if isinstance(image, np.ndarray):
            image /= 255.0

        return image, torch.tensor(label, dtype=torch.int64)
