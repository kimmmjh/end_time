"""Hard decoding metrics for BB code-capacity experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ._bb_loss import pauli_to_xz


def _gf2_product(left: Tensor, right_transposed: Tensor) -> Tensor:
    """Small dense GF(2) product, returned as a boolean tensor."""

    return (
        (
            left.to(dtype=torch.float32)
            @ right_transposed.to(device=left.device, dtype=torch.float32)
        )
        .remainder(2)
        .to(dtype=torch.bool)
    )


@dataclass
class BBShotOutcomes:
    """Per-shot outcomes used for exact aggregation and paired comparison."""

    success: Tensor
    syndrome_converged: Tensor
    flagged_failure: Tensor
    unflagged_logical_failure: Tensor
    pauli_correct: Tensor


@dataclass
class BBAggregateMetrics:
    """Aggregate block-level decoder metrics."""

    samples: int
    logical_accuracy: float
    logical_error_rate: float
    syndrome_convergence: float
    flagged_failure_rate: float
    unflagged_logical_failure_rate: float
    qubit_pauli_accuracy: float
    standard_error: float


def bb_shot_outcomes(
    logits: Tensor,
    syndrome: Tensor,
    pauli: Tensor,
    *,
    hx: Tensor,
    hz: Tensor,
    logicals_x: Tensor,
    logicals_z: Tensor,
) -> BBShotOutcomes:
    """Score hard Pauli corrections by residual stabilizer coset.

    Decoder ``accuracy`` means block logical success.  Matching the exact
    sampled Pauli is only a diagnostic because stabilizer-degenerate
    corrections are equally valid.
    """

    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError(f"Expected logits [B,n,4], got {tuple(logits.shape)}.")
    correction = logits.argmax(dim=-1)
    correction_x, correction_z = pauli_to_xz(correction)
    error_x, error_z = pauli_to_xz(pauli)

    hx = torch.as_tensor(hx, device=logits.device, dtype=torch.bool)
    hz = torch.as_tensor(hz, device=logits.device, dtype=torch.bool)
    logicals_x = torch.as_tensor(logicals_x, device=logits.device, dtype=torch.bool)
    logicals_z = torch.as_tensor(logicals_z, device=logits.device, dtype=torch.bool)

    predicted_syndrome = torch.cat(
        (
            _gf2_product(correction_z, hx.T),
            _gf2_product(correction_x, hz.T),
        ),
        dim=1,
    )
    syndrome_converged = (
        predicted_syndrome == syndrome.to(device=logits.device, dtype=torch.bool)
    ).all(dim=1)

    residual_x = torch.logical_xor(error_x, correction_x)
    residual_z = torch.logical_xor(error_z, correction_z)
    logical_x_bits = _gf2_product(residual_x, logicals_z.T)
    logical_z_bits = _gf2_product(residual_z, logicals_x.T)
    logical_trivial = (
        torch.cat((logical_x_bits, logical_z_bits), dim=1).logical_not().all(dim=1)
    )

    success = syndrome_converged & logical_trivial
    flagged = ~syndrome_converged
    unflagged = syndrome_converged & ~logical_trivial
    pauli_correct = correction == pauli.to(device=correction.device)
    return BBShotOutcomes(
        success=success,
        syndrome_converged=syndrome_converged,
        flagged_failure=flagged,
        unflagged_logical_failure=unflagged,
        pauli_correct=pauli_correct,
    )


def aggregate_bb_outcomes(outcomes: list[BBShotOutcomes]) -> BBAggregateMetrics:
    """Concatenate batches and compute block rates and Bernoulli error bar."""

    if not outcomes:
        raise ValueError("At least one batch of outcomes is required.")
    success = torch.cat([item.success.detach().cpu() for item in outcomes])
    converged = torch.cat([item.syndrome_converged.detach().cpu() for item in outcomes])
    flagged = torch.cat([item.flagged_failure.detach().cpu() for item in outcomes])
    unflagged = torch.cat(
        [item.unflagged_logical_failure.detach().cpu() for item in outcomes]
    )
    pauli_correct = torch.cat(
        [item.pauli_correct.detach().cpu().reshape(-1) for item in outcomes]
    )
    samples = int(success.numel())
    accuracy = float(success.float().mean())
    return BBAggregateMetrics(
        samples=samples,
        logical_accuracy=accuracy,
        logical_error_rate=1.0 - accuracy,
        syndrome_convergence=float(converged.float().mean()),
        flagged_failure_rate=float(flagged.float().mean()),
        unflagged_logical_failure_rate=float(unflagged.float().mean()),
        qubit_pauli_accuracy=float(pauli_correct.float().mean()),
        standard_error=math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / samples),
    )


def paired_success_gain(
    neural: list[BBShotOutcomes], vanilla: list[BBShotOutcomes]
) -> tuple[float, float, int, int]:
    """Return paired neural-minus-vanilla gain, SE, rescues, and harms."""

    neural_success = torch.cat([item.success.detach().cpu() for item in neural]).to(
        torch.float32
    )
    vanilla_success = torch.cat([item.success.detach().cpu() for item in vanilla]).to(
        torch.float32
    )
    if neural_success.shape != vanilla_success.shape:
        raise ValueError("Paired decoder results have different sample counts.")
    difference = neural_success - vanilla_success
    samples = difference.numel()
    gain = float(difference.mean())
    standard_error = (
        float(difference.std(unbiased=True) / math.sqrt(samples))
        if samples > 1
        else 0.0
    )
    rescues = int(((neural_success == 1) & (vanilla_success == 0)).sum())
    harms = int(((neural_success == 0) & (vanilla_success == 1)).sum())
    return gain, standard_error, rescues, harms


__all__ = [
    "BBAggregateMetrics",
    "BBShotOutcomes",
    "aggregate_bb_outcomes",
    "bb_shot_outcomes",
    "paired_success_gain",
]
