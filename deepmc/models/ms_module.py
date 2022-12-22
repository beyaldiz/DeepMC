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


def fk_ts(X_t, topology):
    X_t_glob = X_t.clone()
    for i in range(1, len(topology)):
        X_t_glob[:, i, :] = X_t_glob[:, i, :] + X_t[:, topology[i], :]
    return X_t_glob


class TSLitModule(LightningModule):
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
        topology: list,
        data_dir: str = "data/",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        
        self.save_hyperparameters(logger=False, ignore=["encoder_net", "decoder_net"])

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        ts_data_mean, ts_data_std = torch.tensor(np.load(ts_stat_dir))
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)

        # loss function
        self.criterion = torch.nn.SmoothL1Loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_err = TranslationalError(in_metric="m", out_metric="mm")
        self.val_err = TranslationalError(in_metric="m", out_metric="mm")
        self.test_err = TranslationalError(in_metric="m", out_metric="mm")

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        l_t = self.encoder_net(x)
        return l_t

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        X_t = batch
        X_t_norm = (X_t - self.ts_data_mean) / self.ts_data_std
        l_t = self.encoder_net(X_t_norm)[-1]
        Y_t_norm = self.decoder_net(l_t)[-1].view(X_t.shape)
        Y_t = Y_t_norm * self.ts_data_std + self.ts_data_mean
        X_t_glob, Y_t_glob = (
            fk_ts(X_t, self.hparams.topology),
            fk_ts(Y_t, self.hparams.topology),
        )

        loss = self.criterion(Y_t_glob, X_t_glob)
        return loss, X_t_glob, Y_t_glob

    def training_step(self, batch: Any, batch_idx: int):
        loss, X_t_glob, Y_t_glob = self.step(batch)

        # log train metrics
        err = self.train_err(Y_t_glob, X_t_glob)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/err", err, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_err.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, X_t_glob, Y_t_glob = self.step(batch)

        # log val metrics
        err = self.val_err(Y_t_glob, X_t_glob)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/err", err, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, X_t_glob, Y_t_glob = self.step(batch)

        # log test metrics
        err = self.test_err(Y_t_glob, X_t_glob)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/err", err, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_err.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay


class MCLitModule(LightningModule):
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
        ts_encoder_ckpt_dir: str,
        optimizer: torch.optim.Optimizer,
        topology: list,
        data_dir: str = "data/",
        beta_1: float = 2.0,
        beta_2: float = 20.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["encoder_net", "decoder_net"])

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ts_encoder_net = TSLitModule.load_from_checkpoint(ts_encoder_ckpt_dir)
        self.ts_encoder_net.freeze()

        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        mc_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mc_statistics.npy"
        skinning_weights = data_dir + "/MS_Synthetic_preprocessed/skinning_weights.npy"
        ts_data_mean, ts_data_std = torch.tensor(np.load(ts_stat_dir))
        mc_data_mean, mc_data_std = torch.tensor(np.load(mc_stat_dir))
        skinning_weights = torch.tensor(np.load(skinning_weights)).unsqueeze(-1)
        skinning_weights = torch.tile(skinning_weights, (1, 1, 3))
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)
        self.register_buffer("mc_data_mean", mc_data_mean)
        self.register_buffer("mc_data_std", mc_data_std)
        self.register_buffer("skinning_weights", skinning_weights)

        # loss function
        self.criterion = torch.nn.SmoothL1Loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        self.train_err = TranslationalError(in_metric="m", out_metric="mm")
        self.val_err = TranslationalError(in_metric="m", out_metric="mm")
        self.test_err = TranslationalError(in_metric="m", out_metric="mm")

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        l_c = self.encoder_net(x)
        return l_c

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        X_c, X_t = batch

        ts_mean, ts_std = self.ts_data_mean, self.ts_data_std
        mc_mean, mc_std = self.mc_data_mean, self.mc_data_std
        X_c_norm = (X_c - mc_mean) / mc_std
        X_t_norm = (X_t - ts_mean) / ts_std

        l_t = self.ts_encoder_net.encoder_net(X_t_norm)[-1]
        Y_t_norm = self.ts_encoder_net.decoder_net(l_t)[-1]
        Y_t_norm = Y_t_norm.view(X_t.shape)

        l_c = self.encoder_net(X_c_norm, Y_t_norm, l_t)
        Y_c_norm = self.decoder_net(l_c, Y_t_norm, l_t)
        Y_c_norm = Y_c_norm.view(X_c.shape)

        Y_c = (Y_c_norm * mc_std) + mc_mean
        Y_c_glob = (Y_c * self.skinning_weights).sum(dim=2)

        X_c_glob = (X_c * self.skinning_weights).sum(dim=2)

        reconst_loss1 = self.hparams.beta_1 * self.criterion(Y_c, X_c)
        reconst_loss2 = self.hparams.beta_2 * self.criterion(Y_c_glob, X_c_glob)
        loss = reconst_loss1 + reconst_loss2

        return loss, X_c_glob, Y_c_glob

    def training_step(self, batch: Any, batch_idx: int):
        loss, X_c_glob, Y_c_glob = self.step(batch)

        # log train metrics
        err = self.train_err(Y_c_glob, X_c_glob)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/err", err, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_err.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, X_c_glob, Y_c_glob = self.step(batch)

        # log val metrics
        err = self.val_err(Y_c_glob, X_c_glob)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/err", err, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, X_c_glob, Y_c_glob = self.step(batch)

        # log test metrics
        err = self.test_err(Y_c_glob, X_c_glob)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/err", err, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_err.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # params_decay_enc, params_no_decay_enc = split_parameters(self.encoder_net)
        # params_decay_dec, params_no_decay_dec = split_parameters(self.decoder_net)

        # params_decay = params_decay_enc + params_decay_dec
        # params_no_decay = params_no_decay_enc + params_no_decay_dec

        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }
        # return {
        #     "optimizer": self.hparams.optimizer(
        #         params=[
        #             {"params": params_decay, "weight_decay": 0.0005},
        #             {"params": params_no_decay},
        #         ]
        #     )
        # }


class MOLitModule(LightningModule):
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
        ts_encoder_ckpt_dir: str,
        fk,
        optimizer: torch.optim.Optimizer,
        topology: list,
        data_dir: str = "data/",
        beta_1: float = 2.5,
        beta_2: float = 1.0,
        beta_3: float = 100.0,
        beta_4: float = 5.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["encoder_net", "decoder_net"])

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ts_encoder_net = TSLitModule.load_from_checkpoint(ts_encoder_ckpt_dir)
        self.ts_encoder_net.freeze()
        self.fk = fk

        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        mo_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mo_statistics.npy"
        ts_data_mean, ts_data_std = torch.tensor(np.load(ts_stat_dir))
        mo_data_mean, mo_data_std = torch.tensor(np.load(mo_stat_dir))[:, None, :, None]
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)
        self.register_buffer("mo_data_mean", mo_data_mean)
        self.register_buffer("mo_data_std", mo_data_std)

        # loss function
        self.criterion_rec = torch.nn.MSELoss()
        self.criterion_ver = vertex_loss()
        self.criterion_ang = angle_loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_pos_err = TranslationalError(in_metric="m", out_metric="mm")
        self.train_ang_err = RotationalError()
        self.val_pos_err = TranslationalError(in_metric="m", out_metric="mm")
        self.val_ang_err = RotationalError()
        self.val_err = MeanMetric()
        self.test_pos_err = TranslationalError(in_metric="m", out_metric="mm")
        self.test_ang_err = RotationalError()

        # for logging best so far validation accuracy
        self.val_err_best = MinMetric()

    def forward(self, x: torch.Tensor):
        l_m = self.encoder_net(x)
        return l_m

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_err_best.reset()

    def step(self, batch: Any):
        X_m, X_t = batch

        ts_mean, ts_std = self.ts_data_mean, self.ts_data_std
        mo_mean, mo_std = self.mo_data_mean, self.mo_data_std
        X_m_norm = (X_m - mo_mean) / mo_std
        X_t_norm = (X_t - ts_mean) / ts_std

        l_t = self.ts_encoder_net.encoder_net(X_t_norm)[-1]
        out_ts_dec = self.ts_encoder_net.decoder_net(l_t)[::-1]
        out_ts_dec[0] = out_ts_dec[0].view(X_t.shape) * ts_std + ts_mean

        l_m = self.encoder_net(X_m_norm, out_ts_dec)
        Y_m_norm = self.decoder_net(l_m, out_ts_dec)
        Y_m = Y_m_norm * mo_std + mo_mean

        J_t_glob, J_R_glob = self.fk.forward_from_raw1(Y_m, X_t, world=True)

        J_t_glob_gt, J_R_glob_gt = self.fk.forward_from_raw1(X_m, X_t, world=True)
        J_t_glob_gt, J_R_glob_gt = J_t_glob_gt.detach(), J_R_glob_gt.detach()

        reconst_loss1 = self.criterion_rec(Y_m_norm, X_m_norm)
        reconst_loss2 = self.criterion_rec(Y_m[:, -3:, :], X_m[:, -3:, :])
        reconst_loss3 = self.criterion_rec(J_t_glob, J_t_glob_gt)

        loss = (
            reconst_loss1
            + (
                reconst_loss2 * self.hparams.beta_1
                + reconst_loss3 * self.hparams.beta_2
            )
            * self.hparams.beta_3
        ) * self.hparams.beta_4

        return loss, J_t_glob, J_R_glob, J_t_glob_gt, J_R_glob_gt

    def training_step(self, batch: Any, batch_idx: int):
        loss, J_t_glob, J_R_glob, J_t_glob_gt, J_R_glob_gt = self.step(batch)

        # log train metrics
        pos_error = self.train_pos_err(J_t_glob, J_t_glob_gt)
        angle_error = self.train_ang_err(J_R_glob, J_R_glob_gt)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/pos_err", pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/ang_err", angle_error, on_step=False, on_epoch=True, prog_bar=True
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_pos_err.reset()
        self.train_ang_err.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, J_t_glob, J_R_glob, J_t_glob_gt, J_R_glob_gt = self.step(batch)
        batch_size = J_t_glob.shape[0]

        # log val metrics
        pos_error = self.val_pos_err(J_t_glob, J_t_glob_gt)
        angle_error = self.val_ang_err(J_R_glob, J_R_glob_gt)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/pos_err", pos_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/ang_err", angle_error, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        err = self.val_err.compute()  # get val accuracy from current epoch
        self.val_err_best.update(err)
        self.log(
            "val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True
        )
        self.val_err.reset()
        self.val_pos_err.reset()
        self.val_ang_err.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, J_t_glob, J_R_glob, J_t_glob_gt, J_R_glob_gt = self.step(batch)

        # log test metrics
        pos_error = self.test_pos_err(J_t_glob, J_t_glob_gt)
        angle_error = self.test_ang_err(J_R_glob, J_R_glob_gt)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/pos_err", pos_error, on_step=False, on_epoch=True)
        self.log("test/ang_err", angle_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_pos_err.reset()
        self.test_ang_err.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


class MSLitModule(LightningModule):
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
        ts_encoder_ckpt_dir,
        mc_encoder_ckpt_dir,
        mo_encoder_ckpt_dir,
        fk,
        skin,
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
        self.save_hyperparameters(logger=False, ignore=["encoder_net", "decoder_net"])

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ts_encoder_net = TSLitModule.load_from_checkpoint(ts_encoder_ckpt_dir)
        self.ts_encoder_net.freeze()
        self.mc_encoder_net = MCLitModule.load_from_checkpoint(mc_encoder_ckpt_dir)
        self.mc_encoder_net.freeze()
        self.mo_encoder_net = MOLitModule.load_from_checkpoint(mo_encoder_ckpt_dir)
        self.mo_encoder_net.freeze()
        self.fk = fk
        self.skin = skin

        clean_markers_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/clean_markers_statistics.npy"
        )
        first_rot_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/first_rot_statistics.npy"
        )
        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        mc_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mc_statistics.npy"
        mo_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mo_statistics.npy"
        ts_latent_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/ts_latent_statistics.npy"
        )
        mc_latent_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/mc_latent_statistics.npy"
        )
        mo_latent_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/mo_latent_statistics.npy"
        )
        skinning_weights_dir = (
            data_dir + "/MS_Synthetic_preprocessed/skinning_weights.npy"
        )

        clean_markers_mean, clean_markers_std = torch.tensor(
            np.load(clean_markers_stat_dir), dtype=torch.float32
        )
        first_rot_mean, first_rot_std = torch.tensor(
            np.load(first_rot_stat_dir), dtype=torch.float32
        )
        ts_data_mean, ts_data_std = torch.tensor(
            np.load(ts_stat_dir), dtype=torch.float32
        )
        mc_data_mean, mc_data_std = torch.tensor(
            np.load(mc_stat_dir), dtype=torch.float32
        )
        mo_data_mean, mo_data_std = torch.tensor(
            np.load(mo_stat_dir), dtype=torch.float32
        )[:, None, :, None]
        ts_latent_mean, ts_latent_std = torch.tensor(
            np.load(ts_latent_stat_dir), dtype=torch.float32
        )
        mc_latent_mean, mc_latent_std = torch.tensor(
            np.load(mc_latent_stat_dir), dtype=torch.float32
        )
        mo_latent_mean, mo_latent_std = torch.tensor(
            np.load(mo_latent_stat_dir), dtype=torch.float32
        )
        skinning_weights = torch.tensor(
            np.load(skinning_weights_dir), dtype=torch.float32
        )

        lambda_para = torch.tensor(
            self.encoder_net.lambda_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt1 = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ..., None], dtype=torch.float32
        )

        self.register_buffer("clean_markers_mean", clean_markers_mean)
        self.register_buffer("clean_markers_std", clean_markers_std)
        self.register_buffer("first_rot_mean", first_rot_mean)
        self.register_buffer("first_rot_std", first_rot_std)
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)
        self.register_buffer("mc_data_mean", mc_data_mean)
        self.register_buffer("mc_data_std", mc_data_std)
        self.register_buffer("mo_data_mean", mo_data_mean)
        self.register_buffer("mo_data_std", mo_data_std)
        self.register_buffer("ts_latent_mean", ts_latent_mean)
        self.register_buffer("ts_latent_std", ts_latent_std)
        self.register_buffer("mc_latent_mean", mc_latent_mean)
        self.register_buffer("mc_latent_std", mc_latent_std)
        self.register_buffer("mo_latent_mean", mo_latent_mean)
        self.register_buffer("mo_latent_std", mo_latent_std)
        self.register_buffer("skinning_weights", skinning_weights)

        self.register_buffer("lambda_para", lambda_para)
        self.register_buffer("lambda_para_jt", lambda_para_jt)
        self.register_buffer("lambda_para_jt1", lambda_para_jt1)

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
        self.train_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_joe = RotationalError()
        self.val_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_err = MeanMetric()
        self.test_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.test_joe = RotationalError()
        self.test_mpe = TranslationalError(in_metric="m", out_metric="mm")

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
        l_c_norm, l_m_norm, l_t_norm = self.decoder_net(latent)
        l_c = l_c_norm * self.mc_latent_std + self.mc_latent_mean
        l_t = l_t_norm * self.ts_latent_std + self.ts_latent_mean
        # Check this part?
        l_m_norm0 = l_m_norm[:, 256:]
        l_m0 = l_m_norm0 * self.mo_latent_std + self.mo_latent_mean
        l_m0 = l_m0.view(-1, 112, 16)
        l_m1_norm = l_m_norm[:, :256].view(-1, 64, 4)
        out_first_rot = l_m1_norm * self.first_rot_std + self.first_rot_mean
        out_first_rot = F.normalize(out_first_rot, dim=2)

        out_offset = self.ts_encoder_net.decoder_net(l_t)[::-1]
        # Hard coded
        out_offset[0] = out_offset[0].view(-1, 24, 3)
        out_offset[0] = out_offset[0] * self.ts_data_std + self.ts_data_mean
        res_offset_output_input = (out_offset[0] - self.ts_data_mean) / self.ts_data_std

        out_markerconf = self.mc_encoder_net.decoder_net(
            l_c, res_offset_output_input, l_t
        )
        res_markerconf_all = (out_markerconf * self.mc_data_std) + self.mc_data_mean
        res_markerconf = res_markerconf_all[:, :, :, :3]

        motion_code = l_m0.view(-1, 112, 16)
        out_motion = self.mo_encoder_net.decoder_net(motion_code, out_offset)
        res_motion = (out_motion * self.mo_data_std) + self.mo_data_mean

        res_pos, res_rot = self.fk.forward_from_raw2(
            res_motion, out_offset[0], out_first_rot, world=True, quater=True
        )
        res_markers = self.skin.skinning(
            res_markerconf, self.skinning_weights[None, ...], res_rot, res_pos
        )

        res_first_rot_xform = self.fk.transform_from_quaternion(out_first_rot)
        first_rot_xform = self.fk.transform_from_quaternion(first_rot)

        mrk_pos_loss = self.criterion_huber(
            clean_marker * self.lambda_para, res_markers * self.lambda_para
        )
        skel_pos_loss = self.criterion_huber(
            skeleton_pos * self.lambda_para_jt, res_pos * self.lambda_para_jt
        )
        first_rot_xform_loss = self.criterion_huber(
            first_rot_xform, res_first_rot_xform
        )
        motion_xform_loss = self.criterion_huber(
            transform * self.lambda_para_jt1, res_rot * self.lambda_para_jt1
        )
        offset_latent_loss = self.criterion_huber(of_code, l_t)
        motion_loss = self.criterion_huber(res_motion[..., :-3], motion[..., :-3])
        motion_trans_loss = self.criterion_huber(
            res_motion[:, -3:, :].permute(0, 2, 1), motion[:, -3:, :].permute(0, 2, 1)
        )
        marker_latent_loss = self.criterion_huber(mc_code, l_c)
        marker_config_loss = self.criterion_huber(
            marker_config[:, :, :, :3], res_markerconf
        )
        offset_loss = self.criterion_huber(
            offsets * self.lambda_para_jt, out_offset[0] * self.lambda_para_jt
        )

        loss = (
            mrk_pos_loss * self.hparams.beta_1
            + skel_pos_loss * self.hparams.beta_2
            + first_rot_xform_loss * self.hparams.beta_3
            + motion_xform_loss * self.hparams.beta_4
            + offset_latent_loss * self.hparams.beta_5
            + motion_loss * self.hparams.beta_6
            + motion_trans_loss * self.hparams.beta_7
            + marker_latent_loss * self.hparams.beta_8
            + marker_config_loss * self.hparams.beta_9
            + offset_loss * self.hparams.beta_10
        )

        return (
            loss,
            clean_marker,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log train metrics
        joint_pos_error = self.train_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.train_joe(transform, res_rot)
        marker_pos_error = self.train_mpe(clean_markers, res_markers)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_jpe.reset()
        self.train_joe.reset()
        self.train_mpe.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)
        batch_size = clean_markers.shape[0]

        # log val metrics
        joint_pos_error = self.val_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.val_joe(transform, res_rot)
        marker_pos_error = self.val_mpe(clean_markers, res_markers)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=True
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
        self.val_mpe.reset()

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log test metrics
        joint_pos_error = self.test_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.test_joe(transform, res_rot)
        marker_pos_error = self.test_mpe(clean_markers, res_markers)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/jpe", joint_pos_error, on_step=False, on_epoch=True)
        self.log("test/joe", joint_ori_error, on_step=False, on_epoch=True)
        self.log("test/mpe", marker_pos_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_jpe.reset()
        self.test_joe.reset()
        self.test_mpe.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]


class MSNoEncLitModule(LightningModule):
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
        ts_decoder_net: torch.nn.Module,
        mc_decoder_net: torch.nn.Module,
        mo_decoder_net: torch.nn.Module,
        fk,
        skin,
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
                "encoder_net", "decoder_net", "ts_decoder_net", "mc_decoder_net", "mo_decoder_net"
            ],
        )

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.ts_decoder_net = ts_decoder_net
        self.mc_decoder_net = mc_decoder_net
        self.mo_decoder_net = mo_decoder_net
        self.fk = fk
        self.skin = skin

        clean_markers_stat_dir = (
            data_dir + "/MS_Synthetic_preprocessed/clean_markers_statistics.npy"
        )

        # Hard coded !!!!!
        first_rot_stat_dir = (
            data_dir + "/ours_Synthetic/msalign_first_rot_statistics.npy"
        )
        ts_stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        mc_stat_dir = data_dir + "/MS_Synthetic_preprocessed/mc_statistics.npy"

        # Hard coded !!!!!
        mo_stat_dir = data_dir + "/ours_Synthetic/msalign_mo_statistics.npy"
        skinning_weights_dir = (
            data_dir + "/MS_Synthetic_preprocessed/skinning_weights.npy"
        )

        clean_markers_mean, clean_markers_std = torch.tensor(
            np.load(clean_markers_stat_dir), dtype=torch.float32
        )
        first_rot_mean, first_rot_std = torch.tensor(
            np.load(first_rot_stat_dir), dtype=torch.float32
        )
        ts_data_mean, ts_data_std = torch.tensor(
            np.load(ts_stat_dir), dtype=torch.float32
        )
        mc_data_mean, mc_data_std = torch.tensor(
            np.load(mc_stat_dir), dtype=torch.float32
        )
        mo_data_mean, mo_data_std = torch.tensor(
            np.load(mo_stat_dir), dtype=torch.float32
        )[:, None, :, None]
        skinning_weights = torch.tensor(
            np.load(skinning_weights_dir), dtype=torch.float32
        )

        lambda_para = torch.tensor(
            self.encoder_net.lambda_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ...], dtype=torch.float32
        )
        lambda_para_jt1 = torch.tensor(
            self.encoder_net.lambda_jt_para[None, None, ..., None], dtype=torch.float32
        )

        self.register_buffer("clean_markers_mean", clean_markers_mean)
        self.register_buffer("clean_markers_std", clean_markers_std)
        self.register_buffer("first_rot_mean", first_rot_mean)
        self.register_buffer("first_rot_std", first_rot_std)
        self.register_buffer("ts_data_mean", ts_data_mean)
        self.register_buffer("ts_data_std", ts_data_std)
        self.register_buffer("mc_data_mean", mc_data_mean)
        self.register_buffer("mc_data_std", mc_data_std)
        self.register_buffer("mo_data_mean", mo_data_mean)
        self.register_buffer("mo_data_std", mo_data_std)
        self.register_buffer("skinning_weights", skinning_weights)

        self.register_buffer("lambda_para", lambda_para)
        self.register_buffer("lambda_para_jt", lambda_para_jt)
        self.register_buffer("lambda_para_jt1", lambda_para_jt1)

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
        self.train_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_joe = RotationalError()
        self.val_mpe = TranslationalError(in_metric="m", out_metric="mm")
        self.val_err = MeanMetric()
        self.test_jpe = TranslationalError(in_metric="m", out_metric="mm")
        self.test_joe = RotationalError()
        self.test_mpe = TranslationalError(in_metric="m", out_metric="mm")

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
        l_c, l_m, l_t = self.decoder_net(latent)
        # Check this part?
        l_m0 = l_m[:, 256:]
        l_m0 = l_m0.view(-1, 112, 16)
        l_m1_norm = l_m[:, :256].view(-1, 64, 4)
        out_first_rot = l_m1_norm * self.first_rot_std + self.first_rot_mean
        out_first_rot = F.normalize(out_first_rot, dim=2)

        out_offset = self.ts_decoder_net(l_t)[::-1]
        # Hard coded
        out_offset[0] = out_offset[0].view(-1, 24, 3)
        out_offset[0] = out_offset[0] * self.ts_data_std + self.ts_data_mean
        res_offset_output_input = (out_offset[0] - self.ts_data_mean) / self.ts_data_std

        out_markerconf = self.mc_decoder_net(l_c, res_offset_output_input, l_t)
        res_markerconf_all = (out_markerconf * self.mc_data_std) + self.mc_data_mean
        res_markerconf = res_markerconf_all[:, :, :, :3]

        motion_code = l_m0.view(-1, 112, 16)
        out_motion = self.mo_decoder_net(motion_code, out_offset)
        res_motion = (out_motion * self.mo_data_std) + self.mo_data_mean

        res_pos, res_rot = self.fk.forward_from_raw2(
            res_motion, out_offset[0], out_first_rot, world=True, quater=True
        )
        res_markers = self.skin.skinning(
            res_markerconf, self.skinning_weights[None, ...], res_rot, res_pos
        )

        res_first_rot_xform = self.fk.transform_from_quaternion(out_first_rot)
        first_rot_xform = self.fk.transform_from_quaternion(first_rot)

        mrk_pos_loss = self.criterion_huber(
            clean_marker * self.lambda_para, res_markers * self.lambda_para
        )
        skel_pos_loss = self.criterion_huber(
            skeleton_pos * self.lambda_para_jt, res_pos * self.lambda_para_jt
        )
        first_rot_xform_loss = self.criterion_huber(
            first_rot_xform, res_first_rot_xform
        )
        motion_xform_loss = self.criterion_huber(
            transform * self.lambda_para_jt1, res_rot * self.lambda_para_jt1
        )
        motion_loss = self.criterion_huber(res_motion[..., :-3], motion[..., :-3])
        motion_trans_loss = self.criterion_huber(
            res_motion[:, -3:, :].permute(0, 2, 1), motion[:, -3:, :].permute(0, 2, 1)
        )
        marker_config_loss = self.criterion_huber(
            marker_config[:, :, :, :3], res_markerconf
        )
        offset_loss = self.criterion_huber(
            offsets * self.lambda_para_jt, out_offset[0] * self.lambda_para_jt
        )

        loss = (
            mrk_pos_loss * self.hparams.beta_1
            + skel_pos_loss * self.hparams.beta_2
            + first_rot_xform_loss * self.hparams.beta_3
            + motion_xform_loss * self.hparams.beta_4
            # + offset_latent_loss * self.hparams.beta_5
            + motion_loss * self.hparams.beta_6
            + motion_trans_loss * self.hparams.beta_7
            # + marker_latent_loss * self.hparams.beta_8
            + marker_config_loss * self.hparams.beta_9
            + offset_loss * self.hparams.beta_10
        )

        return (
            loss,
            clean_marker,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log train metrics
        joint_pos_error = self.train_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.train_joe(transform, res_rot)
        marker_pos_error = self.train_mpe(clean_markers, res_markers)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=False
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_jpe.reset()
        self.train_joe.reset()
        self.train_mpe.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)
        batch_size = clean_markers.shape[0]

        # log val metrics
        joint_pos_error = self.val_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.val_joe(transform, res_rot)
        marker_pos_error = self.val_mpe(clean_markers, res_markers)
        self.val_err.update(loss, weight=batch_size)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/jpe", joint_pos_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/joe", joint_ori_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/mpe", marker_pos_error, on_step=False, on_epoch=True, prog_bar=True
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
        self.val_mpe.reset()

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            clean_markers,
            res_markers,
            transform,
            res_rot,
            skeleton_pos,
            res_pos,
        ) = self.step(batch)

        # log test metrics
        joint_pos_error = self.test_jpe(skeleton_pos, res_pos)
        joint_ori_error = self.test_joe(transform, res_rot)
        marker_pos_error = self.test_mpe(clean_markers, res_markers)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/jpe", joint_pos_error, on_step=False, on_epoch=True)
        self.log("test/joe", joint_ori_error, on_step=False, on_epoch=True)
        self.log("test/mpe", marker_pos_error, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_jpe.reset()
        self.test_joe.reset()
        self.test_mpe.reset()

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
