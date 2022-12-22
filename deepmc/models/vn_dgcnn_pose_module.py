from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
import numpy as np

from deepmc.utils.metrics import RotationalError
from deepmc.utils.transform import symmetric_orthogonalization

class VNDGCNNLitModule(LightningModule):
    """Example of LightningModule for template skeleton autoencoder.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "net"
            ],
        )

        self.net = net

        # loss function
        self.criterion = nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_err = RotationalError()
        self.val_err = RotationalError()
        self.test_err = RotationalError()

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        output = self.encoder_net(x)
        return output

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        (
            pc, R_gt
        ) = batch

        R_raw = self.net(pc)
        R = symmetric_orthogonalization(R_raw)
        R = R.view(R_raw.shape[0], 3, 3)

        loss = torch.sqrt(self.criterion(R, R_gt))

        return (
            loss,
            R,
            R_gt
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            R,
            R_gt
        ) = self.step(batch)

        # log train metrics
        ori_error = self.train_err(R, R_gt)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/err", ori_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_err.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            R,
            R_gt
        ) = self.step(batch)

        # log val metrics
        ori_error = self.val_err(R, R_gt)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/err", ori_error, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            R,
            R_gt
        ) = self.step(batch)

        # log test metrics
        ori_error = self.test_err(R, R_gt)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/err", ori_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_err.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "ms_ts.yaml")
    _ = hydra.utils.instantiate(cfg)
