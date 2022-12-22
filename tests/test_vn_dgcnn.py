import math

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pytorch3d.transforms import random_rotation

from deepmc.models.components.vn_dgcnn_pose_net import VN_DGCNN_pose, VN_DGCNN_pose_seg
from deepmc.models.components.vnn import VNMaxPool


@pytest.mark.parametrize("batch_size", [2, 8])
def test_vn(batch_size):
    model = VN_DGCNN_pose().to("cuda:7")

    inp = torch.rand(batch_size, 3, 1024).to("cuda:7")
    out = model(inp).transpose(-1, -2)
    print(out.shape)

    R = random_rotation().to("cuda:7")
    inp_r = R @ inp
    out2 = model(inp_r).transpose(-1, -2)
    out_r = R @ out
    
    print(out_r[0,0,0], out2[0,0,0])
    print(torch.abs(out_r - out2).max())

@pytest.mark.parametrize("batch_size", [1, 8])
def test_vnmaxpool(batch_size):
    model = VNMaxPool(682)

    inp = torch.rand(batch_size, 682, 3, 1024)
    out = model(inp)

    print(out.shape)

    # R = random_rotation()
    # inp_r = R @ inp.unsqueeze(-1)
    # inp_r = inp_r.squeeze()
    # out2 = m_enc(inp_r)
    # out2 = m_dec(out2)
    
    # print(torch.max(out1_r - out2))



def main():
    test_vn(4)
    # test_vnmaxpool(1)

if __name__ == "__main__":
    main()