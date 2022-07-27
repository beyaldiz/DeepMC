from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
import numpy as np

from deepmc.utils.metrics import TranslationalError


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
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        data_dir: str = "data/",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        stat_dir = data_dir + "/MS_Synthetic_preprocessed/ts_statistics.npy"
        self.data_mean, self.data_std = torch.tensor(np.load(stat_dir))

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
        mean, std = self.data_mean.type_as(batch), self.data_std.type_as(batch)
        X_t = batch
        X_t_norm = (X_t - mean) / std
        l_t = self.encoder_net(X_t_norm)[-1]
        Y_t_norm = self.decoder_net(l_t)[-1].view(X_t.shape)
        Y_t = Y_t_norm * std + mean
        X_t_glob, Y_t_glob = fk_ts(X_t, self.hparams.topology), fk_ts(Y_t, self.hparams.topology)
        
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
        self.log("val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True)
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
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        data_dir: str = "data/",
        beta_3: float = 2.0,
        beta_4: float = 20.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["encoder_net, decoder_net"])

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
        
        reconst_loss1 = self.hparams.beta_3 * self.criterion(Y_c, X_c)
        reconst_loss2 = self.hparams.beta_4 * self.criterion(Y_c_glob, X_c_glob)
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
        self.log("val/err_best", self.val_err_best.compute(), on_epoch=True, prog_bar=True)
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
        params_decay_enc, params_no_decay_enc = split_parameters(self.encoder_net)
        params_decay_dec, params_no_decay_dec = split_parameters(self.decoder_net)

        params_decay = params_decay_enc + params_decay_dec
        params_no_decay = params_no_decay_enc + params_no_decay_dec

        return {
            "optimizer": self.hparams.optimizer(params=[{'params': params_decay, 'weight_decay': 0.0005},
                                                        {'params': params_no_decay}])
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "ms_ts.yaml")
    _ = hydra.utils.instantiate(cfg)
