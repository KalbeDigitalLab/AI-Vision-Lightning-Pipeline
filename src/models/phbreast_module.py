from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchmetrics as tm
from lightning import LightningModule


class PHBreastLitModule(LightningModule):
    """LightningModule for Parametrized Hypercomplex Breast Cancer Classification.

    PHBreast training pipeline as described in the paper.
    Source: https://github.com/ispamm/PHBreast/blob/main/training.py
    Reference:
        - arxiv.org/abs/2204.05798

    Parameters
    ----------
    net : torch.nn.Module
        The model module or configuration
    num_classes : int, optional
        Number of output classes, by default 2
    lr : float, optional
        Optimizer learning rate, by default 0.00001
    weight_decay : float, optional
        Optimizer weight decay, by default 0.0005
    optimizer_type : str, optional
        Optimizer type, by default 'adam'
    scheduler_type : Optional[str], optional
        Scheduler type, by default None

    Raises
    ------
    ValueError
        If the optimizer type is not supported.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int = 2,
        lr: float = 0.00001,
        weight_decay: float = 0.0005,
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = None,
    ):
        super().__init__()

        if optimizer_type.lower() not in ['adam']:
            raise ValueError('Optimizer {} is not supported. Only [Adam] is supported.')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = tm.MetricCollection({
            'acc': tm.Accuracy(task='multiclass', num_classes=num_classes, average='macro'),
            'prec': tm.Precision(task='multiclass', num_classes=num_classes, average='macro'),
            'rec': tm.Recall(task='multiclass', num_classes=num_classes, average='macro'),
            'auroc': tm.AUROC(task='multiclass', num_classes=num_classes, average='macro'),
            'f1': tm.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        })

        self.val_metrics = self.train_metrics.clone()

        self.test_metrics = self.train_metrics.clone()

        # for averaging loss across batches
        self.train_mean_loss = tm.MeanMetric()
        self.val_mean_loss = tm.MeanMetric()
        self.test_mean_loss = tm.MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_mean_loss.reset()
        self.val_metrics.reset()

    def model_step(self, batch: Any):
        images_stack, breast_classes, _, _ = batch
        breast_classes = breast_classes.squeeze().long()
        logits = self.forward(images_stack)
        loss = self.criterion(logits, breast_classes)
        preds = F.softmax(logits, dim=1)
        return loss, preds, breast_classes

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_mean_loss(loss)
        metrics = self.train_metrics(preds, targets)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loss_mean', self.train_mean_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/prec', metrics['prec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/rec', metrics['rec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/auroc', metrics['auroc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_mean_loss(loss)
        metrics = self.val_metrics(preds, targets)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss_mean', self.val_mean_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/prec', metrics['prec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/rec', metrics['rec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/auroc', metrics['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_mean_loss(loss)
        metrics = self.test_metrics(preds, targets)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/loss_mean', self.test_mean_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/prec', metrics['prec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/rec', metrics['rec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/auroc', metrics['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization. Normally
        you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        return {'optimizer': optimizer}


if __name__ == '__main__':
    _ = PHBreastLitModule(None, None, None)