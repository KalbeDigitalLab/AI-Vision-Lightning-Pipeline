from typing import Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.components.bodypartxr_parser import VinDrBodyPartXRDataset


class VinDrBodyPartXRDatamodule(LightningDataModule):
    """LightningDataModule for VinDrBodyPartXR Data Pipeline.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

    Parameters
    ----------
    train_dir : str
        Activeloop training dataset directory
    val_dir : str
        Activeloop validation dataset directory
    test_dir : str
        Activeloop testing dataset directory
    input_size : List[int], optional
        Input model size, by default [384, 384]
    batch_size : int, optional
        Number of training batch size, by default 32
    num_workers : int, optional
        Number of worksers to process data, by default 0
    pin_memory : bool, optional
        Enable memory pinning, by default False
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        input_size: Tuple[int, int] = (384, 384),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.50807575], std=[0.20823])])

        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.50807575], std=[0.20823])])

        self.ds_train = train_dir
        self.ds_val = val_dir
        self.ds_test = test_dir

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        """Get number of classes."""
        return 5

    @property
    def train_data(self):
        """Train dataset."""
        return self.data_train

    @property
    def val_data(self):
        """Validation dataset."""
        return self.data_val

    @property
    def test_data(self):
        """Test dataset."""
        return self.data_test

    @property
    def predict_data(self):
        """Prediction dataset."""
        return self.data_predict

    def setup(self, stage: Optional[str] = None):
        """Load the data with specified stage."""
        if stage in ['train', 'fit', None] and self.data_train is None:
            self.data_train = VinDrBodyPartXRDataset(
                dataset_dir=self.ds_train, transform=self.train_transforms)
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
        if stage in ['validation', 'test', 'fit', None]:
            if self.data_val is None:
                self.data_val = VinDrBodyPartXRDataset(
                    dataset_dir=self.ds_val, transform=self.val_transforms)
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = VinDrBodyPartXRDataset(
                    dataset_dir=self.ds_test, transform=self.val_transforms)
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
        if stage == 'predict':
            if self.data_test is None:
                self.data_predict = VinDrBodyPartXRDataset(
                    dataset_dir=self.ds_test, transform=self.val_transforms)
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
