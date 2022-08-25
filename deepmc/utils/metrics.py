import math

import torch
from torchmetrics import Metric


class TranslationalError(Metric):
    higher_is_better = False
    full_state_update = False

    def __init__(self, in_metric: str = "m", out_metric: str = "mm"):
        super().__init__()
        supported_metrics = ["mm", "cm", "dm", "m"]
        assert in_metric in supported_metrics and out_metric in supported_metrics

        self.multiplier = 10.0 ** (
            supported_metrics.index(in_metric) - supported_metrics.index(out_metric)
        )
        self.add_state(
            "sum_mean_translational_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.tensor, targets: torch.tensor):
        assert preds.shape == targets.shape

        num_samples = preds.shape[0]
        sum_mean_translational_error = torch.mean(
            torch.sqrt(torch.sum((preds - targets) ** 2, dim=-1)).view(num_samples, -1),
            dim=-1,
        ).sum()

        self.sum_mean_translational_error += sum_mean_translational_error
        self.num_samples += num_samples

    def compute(self):
        return self.multiplier * (self.sum_mean_translational_error / self.num_samples)


class RotationalError(Metric):
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()

        self.add_state(
            "sum_mean_rotational_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.tensor, targets: torch.tensor):
        assert preds.shape == targets.shape

        num_samples = preds.shape[0]
        R_rel = preds.transpose(-2, -1) @ targets
        angles = torch.rad2deg(
            torch.acos(
                torch.clamp(
                    (torch.diagonal(R_rel, dim1=-1, dim2=-2).sum(dim=-1) - 1.0) / 2,
                    -1.0,
                    1.0,
                )
            )
        )
        sum_mean_rotational_error = torch.mean(
            angles.view(num_samples, -1), dim=-1
        ).sum()

        self.sum_mean_rotational_error += sum_mean_rotational_error
        self.num_samples += num_samples

    def compute(self):
        return self.sum_mean_rotational_error / self.num_samples
