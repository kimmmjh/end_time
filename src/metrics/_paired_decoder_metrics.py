"""Paired hybrid-versus-baseline decoder metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairedDecoderMetrics:
    """Metrics computed from hybrid and baseline decisions on identical shots."""

    num_samples: int
    hybrid_accuracy: float
    baseline_accuracy: float
    rescued: int
    harmed: int
    corrections: int
    rescue_rate: float
    harm_rate: float
    correction_rate: float
    net_gain: float
    paired_standard_error: float

    def as_dict(self) -> dict[str, int | float]:
        """Return checkpoint-friendly primitive values."""

        return asdict(self)


def paired_decoder_metrics(
    final_logits: Tensor,
    true_classes: Tensor,
    baseline_classes: Tensor,
) -> PairedDecoderMetrics:
    """Compare final hybrid and baseline classes on exactly the same samples.

    A rescue is a shot corrected only by the hybrid, while a harm is a shot
    made wrong only by the hybrid.  The paired net gain is therefore
    ``(rescued - harmed) / N`` and is exactly the hybrid accuracy minus the
    baseline accuracy.  Its standard error uses the per-shot paired variable
    in ``{-1, 0, +1}``, rather than treating the two accuracies as independent.
    """

    if final_logits.ndim != 2:
        raise ValueError(
            "Expected final_logits with shape (N, classes), got "
            f"{tuple(final_logits.shape)}."
        )
    if true_classes.ndim != 1 or baseline_classes.ndim != 1:
        raise ValueError(
            "Expected true_classes and baseline_classes with shape (N,), got "
            f"{tuple(true_classes.shape)} and {tuple(baseline_classes.shape)}."
        )

    num_samples = final_logits.shape[0]
    if num_samples < 1:
        raise ValueError("Paired decoder metrics require at least one sample.")
    if (
        true_classes.shape[0] != num_samples
        or baseline_classes.shape[0] != num_samples
    ):
        raise ValueError(
            "Paired inputs must contain the same number of samples: "
            f"{num_samples}, {true_classes.shape[0]}, "
            f"{baseline_classes.shape[0]}."
        )

    device = final_logits.device
    truth = true_classes.to(device=device, dtype=torch.long)
    baseline = baseline_classes.to(device=device, dtype=torch.long)
    hybrid = final_logits.argmax(dim=1)

    hybrid_correct = hybrid == truth
    baseline_correct = baseline == truth
    rescued = int((hybrid_correct & ~baseline_correct).sum().item())
    harmed = int((~hybrid_correct & baseline_correct).sum().item())
    corrections = int((hybrid != baseline).sum().item())
    hybrid_correct_count = int(hybrid_correct.sum().item())
    baseline_correct_count = int(baseline_correct.sum().item())

    hybrid_accuracy = hybrid_correct_count / num_samples
    baseline_accuracy = baseline_correct_count / num_samples
    rescue_rate = rescued / num_samples
    harm_rate = harmed / num_samples
    correction_rate = corrections / num_samples
    net_gain = (rescued - harmed) / num_samples

    if num_samples == 1:
        paired_standard_error = 0.0
    else:
        # For d in {-1, 0, +1}, sum(d^2) is rescued + harmed.
        variance_numerator = rescued + harmed - num_samples * net_gain**2
        variance_numerator = max(0.0, float(variance_numerator))
        paired_standard_error = math.sqrt(
            variance_numerator / (num_samples * (num_samples - 1))
        )

    return PairedDecoderMetrics(
        num_samples=num_samples,
        hybrid_accuracy=hybrid_accuracy,
        baseline_accuracy=baseline_accuracy,
        rescued=rescued,
        harmed=harmed,
        corrections=corrections,
        rescue_rate=rescue_rate,
        harm_rate=harm_rate,
        correction_rate=correction_rate,
        net_gain=net_gain,
        paired_standard_error=paired_standard_error,
    )
