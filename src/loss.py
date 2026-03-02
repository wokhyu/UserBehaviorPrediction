from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def compute_multitask_loss(
    outputs: List[torch.Tensor],
    labels: torch.Tensor,
    loss_fns: List[nn.Module],
    weights: List[float],
) -> torch.Tensor:
    total = torch.tensor(0.0, device=labels.device)
    for idx in range(6):
        total = total + weights[idx] * loss_fns[idx](outputs[idx], labels[:, idx])
    return total
