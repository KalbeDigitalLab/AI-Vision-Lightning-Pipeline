import torchvision.transforms as transforms
import torch
import deeplake

class DeepLakeDataset(Dataset):
    """DeepLake Dataset

    Parameters
    ----------
    dataset: deeplake.Dataset
        DeepLake dataset object
    transform : Optional[torchvision.transforms.Compose], optional
        Data augmentation pipeline, by default None
    """

    def __init__(self, transform: Optional[torchvision.transforms.Compose] = None, dataset = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample for the given index."""

        image = self.dataset.images[idx].numpy()    # Convert the deeplake tensor to numpy array
        label = self.dataset.labels[idx].numpy(fetch_chunks = True).astype(np.int32)

        # Apply transformations
        if self.transform:
          image = self.transform(image)
            # If image is in torch tensor format, convert to PIL Image
        if isinstance(image, np.ndarray):
          image /=255.0

        return image, torch.tensor(label, dtype=torch.int64)