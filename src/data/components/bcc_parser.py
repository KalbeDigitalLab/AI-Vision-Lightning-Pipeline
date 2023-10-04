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

    Parameters
    ----------
    dataset: deeplake.Dataset
        DeepLake dataset object
    transform : Optional[torchvision.transforms.Compose], optional
        Data augmentation pipeline, by default None
    """

    def __init__(self, ds, transform: Optional[torchvision.transforms.Compose] = None):
        self.transform = transform
        if isinstance(ds, str): 
            self.ds = deeplake.load(ds) #path direct
        elif isinstance(ds, deeplake.Dataset): 
            self.ds = ds  #deeplake format
        else: 
            raise TypeError("Invalid Format")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample for the given index."""

        image = self.ds.images[idx].numpy()    # Convert the deeplake tensor to numpy array
        label = self.ds.labels[idx].numpy(fetch_chunks = True).astype(np.int32)

        # Apply transformations
        if self.transform:
          image = self.transform(image)
          
        #normalize to [0, 1]
        if isinstance(image, np.ndarray):
          image /=255.0

        return image, torch.tensor(label, dtype=torch.int64)