"""Calibrated edge-probability loss for neural-weighted MWPM."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EdgeBCELoss(nn.Module):
    """Unweighted edge BCE with optional entropy sharpening.

    Inverse-frequency class weights are intentionally avoided: the logits are
    converted into physical MWPM log-odds at evaluation time, so probability
    calibration matters more than equal positive/negative recall.
    """

    def __init__(self, entropy_weight: float = 0.0) -> None:
        super().__init__()
        if entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative.")
        self.entropy_weight = float(entropy_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Edge logits and targets must match, got {logits.shape} and "
                f"{targets.shape}."
            )
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        if self.entropy_weight == 0.0:
            return bce
        probabilities = torch.sigmoid(logits)
        entropy = -(
            probabilities * F.logsigmoid(logits)
            + (1.0 - probabilities) * F.logsigmoid(-logits)
        ).mean()
        return bce + self.entropy_weight * entropy
