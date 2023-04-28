from typing import Optional

from src.data.components.fiftyone_mammography import FiftyOneVinDrMammography
from src.data.phbreast_datamodule import PHBReastLitDatamodule


class VinDrLitDatamodule(PHBReastLitDatamodule):
    """LightningDataModule for VindrMammo dataset.

    Source: https://physionet.org/content/vindr-mammo/1.0.0/
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def setup(self, stage: Optional[str] = None):
        """Load the data with specified stage."""
        if stage in ['train', 'fit', None] and self.data_train is None:
            self.data_train = FiftyOneVinDrMammography(
                path=self.hparams.data_dir, num_views=self.hparams.num_views,
                stage='train', transform=self.train_transforms)
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
        if stage in ['validation', 'test', 'fit', None]:
            if self.data_val is None:
                self.data_val = FiftyOneVinDrMammography(
                    path=self.hparams.data_dir, num_views=self.hparams.num_views,
                    stage='test', transform=self.val_transforms)
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = FiftyOneVinDrMammography(
                    path=self.hparams.data_dir, num_views=self.hparams.num_views,
                    stage='test', transform=self.val_transforms)
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
        if stage == 'predict':
            if self.data_test is None:
                self.data_predict = FiftyOneVinDrMammography(
                    path=self.hparams.data_dir, num_views=self.hparams.num_views,
                    stage=None, transform=self.val_transforms)
                if len(self.data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')
