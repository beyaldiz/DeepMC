from pathlib import Path

import pytest
import torch

from deepmc.utils.metrics import TranslationalError


@pytest.mark.parametrize("num_batches", [16, 4])
@pytest.mark.parametrize("batch_size", [32, 512])
def test_ms_datamodule(batch_size, num_batches):
    a = torch.rand(num_batches, batch_size, 24, 3)
    b = torch.rand(num_batches, batch_size, 24, 3)
    gt = torch.sqrt(torch.sum((a - b) ** 2, dim=-1)).mean(dim=-1).mean()

    metric = TranslationalError()
    for i in range(num_batches):
        metric.update(a[i], b[i])
    
    assert torch.allclose(metric.compute(), gt)

    metric.reset()

    for i in range(num_batches):
        metric.update(a[i], b[i])
    
    assert torch.allclose(metric.compute(), gt)
