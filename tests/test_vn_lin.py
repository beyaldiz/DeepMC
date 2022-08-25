import math

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from deepmc.models.components.vnn import VNLinearLeakyReLU, VNLinear, VNBatchNorm, VNLeakyReLU
from deepmc.models.components.vn_ms_net import VN_Marker_enc, VN_Marker_dec
from pytorch3d.transforms.rotation_conversions import random_rotation
from deepmc.utils.MoCap_Solver.utils import angle_loss


# @pytest.mark.parametrize("in_channels", [1])
# @pytest.mark.parametrize("out_channels", [16, 128])
# def test_vn(in_channels, out_channels):
#     model = VNLinear(64 * 56, 2048 // 3)
#     bn = VNBatchNorm(2048 // 3, dim=3)
#     relu = VNLeakyReLU(2048 // 3, negative_slope=0.5)
#     inp1 = torch.rand(128, 64, 56, 3)
#     inp1 = inp1.view(inp1.shape[0], -1, inp1.shape[3])
#     out1 = model(inp1)
#     out2 = bn(out1)
#     out3 = relu(out2)
#     print(out3.shape)

@pytest.mark.parametrize("batch_size", [16, 128])
def test_vn_marker_enc(batch_size):
    m_enc = VN_Marker_enc()
    m_dec = VN_Marker_dec()

    inp = torch.rand(batch_size, 64, 56, 3)
    out1 = m_enc(inp)
    o1, o2, o3 = m_dec(out1)

    print(o1.shape, o2.shape, o3.shape)

    # R = random_rotation()
    # inp_r = R @ inp.unsqueeze(-1)
    # inp_r = inp_r.squeeze()
    # out2 = m_enc(inp_r)
    # out2 = m_dec(out2)
    
    # print(torch.max(out1_r - out2))



def main():
    # test_vn(1, 16)
    test_vn_marker_enc(16)

if __name__ == "__main__":
    main()