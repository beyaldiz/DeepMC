from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
import numpy as np

from deepmc.utils.metrics import TranslationalError, RotationalError
from deepmc.utils.MoCap_Solver.utils import (
    vertex_loss,
    angle_loss,
    HuberLoss,
    Criterion_EE,
)
from deepmc.utils.transform import symmetric_orthogonalization

class MSRootLitModule(LightningModule):
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
        encoder_net: torch.nn.Module,
        decoder_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        topology: list,
        data_dir: str = "data/",
        beta_1: float = 20.0,
        beta_2: float = 50.0,
        beta_3: float = 1000.0,
        beta_4: float = 1.0,
        beta_5: float = 2.0,
        beta_6: float = 10.0,
        beta_7: float = 5000.0,
        beta_8: float = 1.0,
        beta_9: float = 100.0,
        beta_10: float = 100.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "encoder_net", "decoder_net"
            ],
        )

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        # loss function
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_huber = HuberLoss(100)
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_ee = Criterion_EE(1, torch.nn.MSELoss())
        self.criterion_ang = angle_loss()
        self.criterion_ver = vertex_loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.train_joe = RotationalError()
        self.val_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_joe = RotationalError()
        self.val_err = MeanMetric()
        self.test_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.test_joe = RotationalError()

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
            raw_marker,
            clean_marker,
            skeleton_pos,
            motion,
            offsets,
            marker_config,
            first_rot,
            of_code,
            mc_code,
            transform,
        ) = batch

        latent = self.encoder_net(raw_marker)
        root_R_raw, root_t = self.decoder_net(latent)
        root_R = symmetric_orthogonalization(root_R_raw)
        root_R = root_R.view(*root_R_raw.shape[:-2], 3, 3)

        root_R_gt = transform[:, :, 0]
        root_t_gt = skeleton_pos[:, :, 0]

        rot_loss = self.criterion_huber(root_R, root_R_gt)
        tr_loss = self.criterion_huber(root_t, root_t_gt)

        loss = (
            tr_loss
            + rot_loss
        )

        return (
            loss,
            root_R,
            transform[:, :, 0],
            root_t,
            skeleton_pos[:, :, 0]
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            root_R,
            root_R_gt,
            root_t,
            root_t_gt
        ) = self.step(batch)

        # log train metrics
        joint_pos_error = self.train_jpe(root_t, root_t_gt)
        joint_ori_error = self.train_joe(root_R, root_R_gt)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_jpe.reset()
        self.train_joe.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            root_R,
            root_R_gt,
            root_t,
            root_t_gt
        ) = self.step(batch)
        batch_size = root_R.shape[0]

        # log val metrics
        joint_pos_error = self.val_jpe(root_t, root_t_gt)
        joint_ori_error = self.val_joe(root_R, root_R_gt)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()
        self.val_jpe.reset()
        self.val_joe.reset()

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            root_R,
            root_R_gt,
            root_t,
            root_t_gt
        ) = self.step(batch)

        # log test metrics
        joint_pos_error = self.test_jpe(root_t, root_t_gt)
        joint_ori_error = self.test_joe(root_R, root_R_gt)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/jpe", joint_pos_error, on_step=False, on_epoch=True)
        self.log("test/joe", joint_ori_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_jpe.reset()
        self.test_joe.reset()

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
