import math

from pathlib import Path

import pytest
import torch

from deepmc.utils.metrics import TranslationalError, RotationalError
from deepmc.utils.MoCap_Solver.utils import angle_loss
from pytorch3d.transforms.rotation_conversions import random_rotations, matrix_to_axis_angle


@pytest.mark.parametrize("num_batches", [16, 4])
@pytest.mark.parametrize("batch_size", [32, 512])
def test_translation_metric(batch_size, num_batches):
    a = torch.rand(num_batches, batch_size, 24, 3)
    b = torch.rand(num_batches, batch_size, 24, 3)
    gt = torch.sqrt(torch.sum((a - b) ** 2, dim=-1)).mean(dim=-1).mean()

    metric = TranslationalError("m", "m")
    for i in range(num_batches):
        metric.update(a[i], b[i])
    
    assert torch.allclose(metric.compute(), gt)

    metric.reset()

    for i in range(num_batches):
        metric.update(a[i], b[i])
    
    assert torch.allclose(metric.compute(), gt)

    a = torch.rand(num_batches, batch_size, 56, 24, 3)
    b = torch.rand(num_batches, batch_size, 56, 24, 3)
    gt = torch.sqrt(torch.sum((a - b) ** 2, dim=-1)).mean()

    metric = TranslationalError("m", "m")
    for i in range(num_batches):
        metric.update(a[i], b[i])
    
    assert torch.allclose(metric.compute(), gt)

    metric.reset()

    for i in range(num_batches):
        metric(a[i], b[i])
    
    assert torch.allclose(metric.compute(), gt)


@pytest.mark.parametrize("num_batches", [16, 4])
@pytest.mark.parametrize("batch_size", [32, 512])
def test_rotation_metric(batch_size, num_batches):
    a = random_rotations(batch_size * num_batches).view(num_batches, batch_size, 3, 3)
    b = random_rotations(batch_size * num_batches).view(num_batches, batch_size, 3, 3)
    angle_crit = angle_loss()
    R_rel = a.transpose(-2, -1) @ b
    axis_angle = matrix_to_axis_angle(R_rel)
    angle_diff = torch.norm(axis_angle, p=2, dim=-1)
    angle_diff[angle_diff > math.pi] -= 2*math.pi
    angle_diff = torch.abs(angle_diff) / math.pi * 180.0
    gt1 = torch.mean(angle_diff)
    gt2 = angle_crit(a, b)
    metric = RotationalError()
    for i in range(num_batches):
        metric(a[i], b[i])
    assert torch.allclose(metric.compute(), gt1)
    assert torch.allclose(metric.compute(), gt2)

