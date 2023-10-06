from typing import Tuple, Optional
import deeplake
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from typing import Callable
from src.data.components.bcc_parser import DeepLakeDataset
from torchvision.transforms import transforms


class DeepLakeLitDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for DeepLake Dataset

    This Lightning DataModule class is designed for handling DeepLake datasets
    within a PyTorch Lightning-based machine learning pipeline. It provides
    data loaders for training, validation, and testing.

    Parameters
    ----------
    train_dir : str
        The path to the directory containing the training dataset.
    val_dir : str
        The path to the directory containing the validation dataset.
    test_dir : str
        The path to the directory containing the testing dataset.
    input_size : Tuple[int, int], optional
        The desired input size for the images, by default [600, 500].
    batch_size : int, optional
        The batch size for data loaders, by default 64.
    num_workers : int, optional
        The number of CPU workers to use for data loading, by default 0.
    pin_memory : bool, optional
        Whether to use pinned memory for faster GPU data transfer, by default False.

    Attributes
    ----------
    num_classes : int
        The number of classes in the dataset.
        Class: normal, benign, malignant

    Methods
    -------
    setup(stage: Optional[str] = None)
        Load the data for the specified stage (train, validation, test, or predict).

    train_dataloader()
        Get the training data loader.
    
    val_dataloader()
        Get the validation data loader.

    test_dataloader()
        Get the test data loader.
    """
    def __init__(
        self,
        train_dir,
        val_dir,
        test_dir,
        input_size: Tuple[int, int] = [600, 500],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.RandomRotation(20),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0], std=[1])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0], std=[1])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0], std=[1])
        ])
        
        self.ds_train = train_dir
        self.ds_val = val_dir
        self.ds_test = test_dir
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_clasess(self):
      return 3

    def setup(self, stage: Optional[str] = None):
        if stage in ['train', 'fit', None] and self.data_train is None:
            self.data_train = DeepLakeDataset(
                ds = self.ds_train, 
                transform = self.train_transform
                )
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
            
        if stage in ['validation', 'fit', None]:
            if self.data_val is None:
                self.data_val = DeepLakeDataset(
                    ds = self.ds_val, 
                    transform=self.val_transform
                    )
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = DeepLakeDataset(
                    ds = self.ds_test, 
                    transform = self.test_transform
                    )
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
                
        if stage == 'predict':
            if self.data_test is None:
                self.data_predict = DeepLakeDataset(
                    ds = self.ds_test, 
                    transform=self.test_transform
                    )
                if len(self.data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )