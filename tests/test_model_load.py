import os

import pytest
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict, DictConfig
import pyrootutils
import torch

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from deepmc.tasks import eval_task, train_task
from deepmc.models.ms_module import TSLitModule

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def test_model_load(cfg: DictConfig) -> None:
    # print(cfg.ckpt_path)
    print(hydra.utils.get_class(cfg.datamodule._target_))
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # model2 = model.load_from_checkpoint(cfg.ckpt_path)
    # model3 = TSLitModule.load_from_checkpoint(cfg.ckpt_path)
    # model.eval()
    # model2.eval()
    # model3.eval()
    # inp = torch.rand(512, 24, 3)
    # out = model(inp)
    # out2 = model2(inp)
    # out3 = model3(inp)
    # assert torch.allclose(out2[-1], out3[-1])
    for m in model.modules():
        print(m)

    # print(loaded_model.encoder_net)


if __name__ == "__main__":
    test_model_load()