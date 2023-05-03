from typing import List, Optional

from src.data.components.fiftyone_mammography import VinDrMammographyDataset
from src.data.phbreast_datamodule import PHBReastLitDatamodule


class VinDrLitDatamodule(PHBReastLitDatamodule):
    """LightningDataModule for VindrMammo dataset.

    Source: https://physionet.org/content/vindr-mammo/1.0.0/
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

    Parameters
    ----------
    data_dir : str, optional
        FiftyOne dataset directory, by default 'data/'
    output_type: str, optional
        Output prediction head type, e.g single or multiple head. By default 'multiple'
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
        output_type: str = 'multiple',
        num_views: int = 2,
        num_classes: int = 2,
        input_size: List[int] = [600, 500],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        if output_type not in ['single', 'multiple']:
            raise ValueError('Unsupported output type {output_type}. Only [single. multiple] are supported')

        super().__init__(data_dir, num_views, num_classes, input_size, batch_size, num_workers, pin_memory)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        """Load the data with specified stage."""
        constructor_args = {
            'path': self.hparams.data_dir,
            'num_views': self.hparams.num_views,
            'transform': self.train_transforms,
        }
        constructor_args['output_type'] = self.hparams.output_type

        if stage in ['train', 'fit', None] and self.data_train is None:
            constructor_args['stage'] = 'train'
            self.data_train = VinDrMammographyDataset(**constructor_args)
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
        if stage in ['validation', 'test', 'fit', None]:
            constructor_args['stage'] = 'test'
            constructor_args['transform'] = self.val_transforms
            if self.data_val is None:
                self.data_val = VinDrMammographyDataset(**constructor_args)
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = VinDrMammographyDataset(**constructor_args)
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
        if stage == 'predict':
            if self.data_test is None:
                constructor_args['stage'] = None
                constructor_args['transform'] = self.val_transforms
                self.data_predict = VinDrMammographyDataset(**constructor_args)
                if len(self.data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')
