import math

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmc.models.components.MoCap_Solver import Marker_enc, Marker_dec_root
from deepmc.models.components.vn_ms_net import VN_Marker_enc, VN_Marker_dec_root
from deepmc.models.components.vnn import VNLinear
from deepmc.utils.transform import symmetric_orthogonalization
from pytorch3d.transforms import random_rotation

def test_marker():
    m_enc = VN_Marker_enc()
    m_dec = VN_Marker_dec_root()

    inp1 = torch.rand(128, 64, 56, 3)
    out1 = m_enc(inp1)
    R1, t1 = m_dec(out1)
    R1 = symmetric_orthogonalization(R1)
    
    rand_rot = random_rotation()

    inp2 = rand_rot @ inp1.unsqueeze(-1)
    inp2 = inp2.squeeze()
    out2 = m_enc(inp2)
    R2, t2 = m_dec(out2)
    R2 = symmetric_orthogonalization(R2)

    R1_rot = rand_rot @ R1
    t1_rot = rand_rot @ t1.unsqueeze(-1)
    t1_rot = t1_rot.squeeze()

    print(torch.mean(torch.abs(R1_rot - R2)))
    print(torch.mean(torch.abs(t1_rot - t2)))


def main():
    test_marker()

if __name__ == "__main__":
    main()