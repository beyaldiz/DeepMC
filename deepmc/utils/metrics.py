import torch
from torchmetrics import Metric


class TranslationalError(Metric):
    higher_is_better = False
    full_state_update = False
    def __init__(self, in_metric: str = "m", out_metric: str = "mm"):
        super().__init__()
        supported_metrics = ["mm", "cm", "dm", "m"]
        self.multiplier = 10.0 ** (supported_metrics.index(in_metric) - supported_metrics.index(out_metric))
        self.add_state("sum_mean_translational_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.tensor, targets: torch.tensor):
        assert preds.shape == targets.shape

        sum_mean_translational_error = torch.mean(torch.sqrt(torch.sum((preds-targets) ** 2, dim=2)), dim=1).sum()
        num_samples = preds.shape[0]

        self.sum_mean_translational_error += sum_mean_translational_error
        self.num_samples += num_samples
    
    def compute(self):
        return self.multiplier * (self.sum_mean_translational_error / self.num_samples)

