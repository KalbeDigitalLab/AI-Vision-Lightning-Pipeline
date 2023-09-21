from typing import Tuple, Optional
import deeplake
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from typing import Callable
from src.data.components.bcc_parser import DeepLakeDataset

class DeepLakeLitDataModule(LightningDataModule):
    def __init__(
        self,
        input_size: Tuple[int, int] = [600, 500],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def num_clasess(self):
      """get number of classes"""
      return 3

    def setup(self, stage: Optional[str] = None):
        """Load the data with specified stage."""
        if stage in ['train', 'fit', None] and self.data_train is None:
            self.data_train = DeepLakeDataset(
                    dataset = ds_train, transform=train_tform)
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
        if stage in ['validation', 'fit', None]:
            if self.data_val is None:
                self.data_val = DeepLakeDataset(
                    dataset=ds_val, transform=val_tform)
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = DeepLakeDataset(
                    dataset=ds_test, transform=test_tform)
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
        if stage == 'predict':
            if self.data_test is None:
                self.data_predict = DeepLakeDataset(
                    dataset=ds_val, transform=val_tform)
                if len(self.data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
           # collate_fn=train_tform,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
           # collate_fn = val_tform,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )