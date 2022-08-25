import math

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from deepmc.models.components.vnn import VNStdFeature
from pytorch3d.transforms.rotation_conversions import random_rotation


class VNInvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_features
    ):
        super().__init__()
        self.inv_layer = VNStdFeature(in_channels)
        self.linear = nn.Linear(3 * in_channels * num_features, 1)
    
    def forward(self, x):
        out, _ = self.inv_layer(x)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)

        return out


@pytest.mark.parametrize("in_channels", [16, 128])
@pytest.mark.parametrize("num_features", [16, 32])
def test_vn(in_channels, num_features):
    model = VNInvNet(in_channels, num_features)
    inp1 = torch.rand(2, in_channels, 3, num_features)
    tmp = inp1.clone().permute(0, 1, 3, 2).unsqueeze(-1)
    R = random_rotation()
    inp2 = R @ tmp
    inp2 = inp2.squeeze().permute(0, 1, 3, 2)
    out1 = model(inp1)
    out2 = model(inp2)

    assert torch.allclose(out1, out2)

def main():
    test_vn(16, 16)

if __name__ == "__main__":
    main()