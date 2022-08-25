import torch
from torch import nn
import numpy as np

from deepmc.models.components.vnn import VNLinearAndLeakyReLU


class VNNet(nn.Module):
    def __init__(
        self,
        skinning_weights: np.ndarray,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
