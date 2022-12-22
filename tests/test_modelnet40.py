from pathlib import Path

import pytest
import torch

from deepmc.datamodules.components.ModelNet40 import ModelNet40Alignment

def test_modelnet():
    dataset = ModelNet40Alignment()
    pc, R = dataset[0]
    print(pc.shape)
    print(R.shape)


def main():
    test_modelnet()


if __name__ == "__main__":
    main()
