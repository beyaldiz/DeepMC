import math

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmc.models.components.MoCap_Solver import MO_enc, MO_dec

def test_mo():
    m_enc = MO_enc()
    m_dec = MO_dec()

    # for layer in m_dec.layers:
    #     print(layer)

    inp = torch.arange(12).view(1, 3, 4).to(torch.float32)
    print(inp)
    upsampler = nn.Upsample(scale_factor=2, mode='linear')
    out = upsampler(inp)
    print(out)


def main():
    test_mo()

if __name__ == "__main__":
    main()