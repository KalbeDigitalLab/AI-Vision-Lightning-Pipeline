import torchvision 
import torch
import deeplake
from torch.utils.data import Dataset
from typing import Any, Optional, Tuple
import numpy as np
import hub
from tests.bc_dummy_dataset import (dummy_dataset, 
                                    dummy_train_dataset, 
                                    dummy_val_dataset, 
                                    dummy_test_dataset
                                    )


class DeepLakeDataset(Dataset):
 """DeepLake Dataset

    This class is a PyTorch Dataset wrapper for DeepLake datasets, allowing you
    to easily integrate them into your PyTorch-based machine learning pipelines.

    Parameters
    ----------
    ds : str or deeplake.Dataset
        The DeepLake dataset object or the path to the dataset directory.
        If `ds` is a string, it should be the path to the DeepLake dataset directory.
        If `ds` is a deeplake.Dataset object, it represents the dataset directly.
    transform : Optional[torchvision.transforms.Compose], optional
        A data augmentation pipeline to apply to the images, by default None.

    Raises
    ------
    TypeError
        If the `ds` parameter is neither a valid DeepLake dataset object nor a string.

    Attributes
    ----------
    transform : Optional[torchvision.transforms.Compose]
        A data augmentation pipeline for transforming the images.

    Methods
    -------
    __len__()
        Get the number of samples in the dataset.

    __getitem__(idx)
        Get a sample for the given index.
    """

    def __init__(self, ds, transform: Optional[torchvision.transforms.Compose] = None):
        """
        Initialize a DeepLakeDataset object.

        Parameters
        ----------
        ds : str or deeplake.Dataset
            The DeepLake dataset object or the path to the dataset directory.
            If `ds` is a string, it should be the path to the DeepLake dataset directory.
            If `ds` is a deeplake.Dataset object, it represents the dataset directly.
        transform : Optional[torchvision.transforms.Compose], optional
            A data augmentation pipeline to apply to the images, by default None.

        Raises
        ------
        TypeError
            If the `ds` parameter is neither a valid DeepLake dataset object nor a string.
        """
        self.transform = transform
        if isinstance(ds, str): 
            self.ds = deeplake.load(ds) #path direct
        elif isinstance(ds, deeplake.Dataset): 
            self.ds = ds  #deeplake format
        else: 
            raise TypeError("Invalid Format")

    def __len__(self) -> int:
        """Get the number of samples in the dataset.
        
        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample for the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image and its corresponding label as PyTorch tensors.
        """

        image = self.ds.images[idx].numpy()    # Convert the deeplake tensor to numpy array
        label = self.ds.labels[idx].numpy(fetch_chunks = True).astype(np.int32)

        # Apply transformations
        if self.transform:
          image = self.transform(image)
          
        #normalize to [0, 1]
        if isinstance(image, np.ndarray):
          image /=255.0

        return image, torch.tensor(label, dtype=torch.int64)