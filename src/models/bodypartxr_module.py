from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics as tm
from lightning import LightningModule


class BodyPartXRLitModule(LightningModule):
    """LightningModule for Body Part X-ray Classification.

    Parameters
    ----------
    net : torch.nn.Module
        The model module or configuration
    num_classes : int, optional
        Number of output classes, by default 5
    lr : float, optional
        Optimizer learning rate, by default 0.00001

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int = 5,
        lr: float = 0.00001
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = tm.Accuracy(
            task='multiclass', num_classes=num_classes)

        self.val_metrics = tm.MetricCollection({
            'acc': tm.Accuracy(task='multiclass', num_classes=num_classes),
            'prec': tm.Precision(task='multiclass', num_classes=num_classes),
            'rec': tm.Recall(task='multiclass', num_classes=num_classes),
            'auroc': tm.AUROC(task='multiclass', num_classes=num_classes),
            'f1': tm.F1Score(task='multiclass', num_classes=num_classes),
        })  # type: ignore

        self.test_metrics = self.val_metrics.clone()

    def reset_metrics(self):
        self.train_acc.reset()
        self.val_metrics.reset()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_metrics()

    def model_step(self, batch: Any):
        images, targets = batch
        targets = targets.squeeze().long()  # convert to 1D
        logits = self.forward(images)
        loss = self.criterion(logits, targets)
        preds = F.softmax(logits, dim=1)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        acc = self.train_acc(preds, targets)
        self.log('train/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_start(self):
        self.reset_metrics()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        metrics = self.val_metrics(preds, targets)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', metrics['acc'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/prec', metrics['prec'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/rec', metrics['rec'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/auroc', metrics['auroc'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/f1', metrics['f1'],
                 on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        metrics = self.test_metrics(preds, targets)
        self.log('test/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('test/acc', metrics['acc'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/prec', metrics['prec'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/rec', metrics['rec'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/auroc', metrics['auroc'],
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/f1', metrics['f1'],
                 on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure the optimizer and scheduler to use.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer
