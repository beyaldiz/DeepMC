from pathlib import Path

import pytest
import torch

from deepmc.datamodules.ms_datamodule import TSDataModule


@pytest.mark.parametrize("batch_size", [32, 512])
def test_ms_datamodule(batch_size):
    data_dir = "data/"

    dm = TSDataModule(data_dir=data_dir, batch_size=batch_size)

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MS_Synthetic_preprocessed").exists()
    assert Path(data_dir, "MS_Synthetic_preprocessed", "train_sample_data").exists()
    assert Path(data_dir, "MS_Synthetic_preprocessed", "test_sample_data").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, mean, std = batch
    assert type(x) is torch.Tensor
    assert type(mean) is torch.Tensor and type(std) is torch.Tensor
    assert x.dtype == torch.float32
    assert mean.dtype == torch.float32 and std.dtype == torch.float32
    assert x.shape[0] == batch_size
    assert mean.shape == x[0].shape and std.shape == x[0].shape
