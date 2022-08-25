import os

import pytest
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict, DictConfig
import pyrootutils
import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from deepmc.tasks import eval_task, train_task
from deepmc.models.ms_module import TSLitModule

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def fk_ts(X_t, topology):
    X_t_glob = X_t.copy()
    for i in range(1, len(topology)):
        X_t_glob[:, i, :] = X_t_glob[:, i, :] + X_t[:, topology[i], :]
    return X_t_glob



@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def test_model_load(cfg: DictConfig) -> None:
    topology = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    model = TSLitModule.load_from_checkpoint(cfg.ckpt_path)
    model.eval()
    data = np.load("data/MS_Synthetic_preprocessed/test_sample_data/77_77_21_poses_0.npz")
    skel = data["J"][None, ...]
    skel = skel - skel[:, topology, :]
    skel = np.concatenate([skel, skel], axis=0)
    mean, std = np.load("data/MS_Synthetic_preprocessed/ts_statistics.npy")
    skel = (skel - mean) / std
    inp = torch.from_numpy(skel).float()
    print(inp.shape)
    out = model.encoder_net(inp)[-1]
    out = model.decoder_net(out)[-1].view(2, 24, 3)
    out = out.detach().numpy()
    out = out * std + mean
    out = fk_ts(out, topology)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter3D(out[0, :, 0], out[0, :, 1], out[0, :, 2])
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.savefig("test.png")



if __name__ == "__main__":
    test_model_load()