from typing import List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.components.phbreast_dataset_parser import PHBreastZIPDataset


class PHBReastLitDatamodule(LightningDataModule):
    """LightningDataModule for default Parametrized Hypercomplex Breast Cancer dataset.

    Source: https://github.com/ispamm/PHBreast
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

    Parameters
    ----------
    data_dir : str, optional
        FiftyOne dataset directory, by default 'data/'
    num_views : int, optional
        Number of mmaography views, by default 2
    num_classes : int, optional
        Number of output classes, by default 2
    input_size : List[int], optional
        Input model size, by default [600, 500]
    batch_size : int, optional
        Number of training batch size, by default 64
    num_workers : int, optional
        Number of worksers to process data, by default 0
    pin_memory : bool, optional
        Enable memory pinning, by default False
    """

    def __init__(
        self,
        data_dir: str = 'data/',
        num_views: int = 2,
        num_classes: int = 2,
        input_size: List[int] = [600, 500],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose([
            transforms.Resize(tuple(input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((-25, 25)),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(tuple(input_size)),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    @property
    def num_views(self):
        """Get number of input views/sides."""
        return self.hparams.num_views

    @property
    def num_classes(self):
        """Get number of classes."""
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None):
        """Load the data with specified stage."""
        if stage in ['train', 'fit', None] and self.data_train is None:
            self.data_train = PHBreastZIPDataset(
                path=self.hparams.data_dir, num_views=self.hparams.num_views,
                stage='train', transform=self.train_transforms)
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
        if stage in ['validation', 'test', 'fit', None]:
            if self.data_val is None:
                self.data_val = PHBreastZIPDataset(
                    path=self.hparams.data_dir, num_views=self.hparams.num_views,
                    stage='validation', transform=self.val_transforms)
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = PHBreastZIPDataset(
                    path=self.hparams.data_dir, num_views=self.hparams.num_views,
                    stage='validation', transform=self.val_transforms)
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
        if stage == 'predict':
            if self.data_test is None:
                self.data_predict = PHBreastZIPDataset(
                    path=self.hparams.data_dir, num_views=self.hparams.num_views,
                    stage=None, transform=self.val_transforms)
                if len(self.data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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
