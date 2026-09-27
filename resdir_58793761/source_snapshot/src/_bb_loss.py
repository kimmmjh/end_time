"""Degeneracy-aware objectives for the BB code-capacity decoder.

The neural BP model returns one four-state Pauli belief per data qubit.  A
sampled physical error is not a unique decoding target: multiplying a valid
correction by a stabilizer gives an equally valid correction.  The primary
terms below are factorized syndrome/logical-parity surrogates that accept this
degeneracy in the deterministic limit.  A small Pauli cross-entropy term is
retained as an optimization aid, but is not the decoder's main objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class BBLossOutput:
    """Individual loss terms, including the differentiable total."""

    total: Tensor
    syndrome: Tensor
    logical: Tensor
    pauli: Tensor
    deep_supervision: Tensor


def pauli_to_xz(pauli: Tensor) -> tuple[Tensor, Tensor]:
    """Convert ``I=0, X=1, Y=2, Z=3`` labels to binary X/Z parts."""

    x = (pauli == 1) | (pauli == 2)
    z = (pauli == 2) | (pauli == 3)
    return x, z


def _odd_parity_probability(
    bit_probability: Tensor,
    support: Tensor,
) -> Tensor:
    """Probability that each supported parity is odd.

    ``bit_probability`` has shape ``[B,n]`` and ``support`` has shape
    ``[m,n]``.  This is the exact parity marginal under the factorized qubit
    beliefs and remains fully differentiable.
    """

    if bit_probability.ndim != 2 or support.ndim != 2:
        raise ValueError("bit_probability and support must be rank-two tensors.")
    if bit_probability.shape[1] != support.shape[1]:
        raise ValueError(
            "Qubit dimension mismatch: "
            f"{bit_probability.shape[1]} != {support.shape[1]}."
        )

    signed = 1.0 - 2.0 * bit_probability
    factors = torch.where(
        support.to(device=bit_probability.device, dtype=torch.bool).unsqueeze(0),
        signed.unsqueeze(1),
        torch.ones((), device=bit_probability.device, dtype=bit_probability.dtype),
    )
    return 0.5 * (1.0 - factors.prod(dim=-1))


class DegeneracyAwareBPLoss(nn.Module):
    """Syndrome/coset objective with a small physical-Pauli auxiliary term."""

    def __init__(
        self,
        *,
        hx: Tensor,
        hz: Tensor,
        logicals_x: Tensor,
        logicals_z: Tensor,
        syndrome_weight: float = 1.0,
        logical_weight: float = 1.0,
        pauli_weight: float = 0.1,
        deep_supervision_weight: float = 0.2,
        probability_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        for name, value in (
            ("syndrome_weight", syndrome_weight),
            ("logical_weight", logical_weight),
            ("pauli_weight", pauli_weight),
            ("deep_supervision_weight", deep_supervision_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if not 0.0 < probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie in (0, 0.5).")

        self.register_buffer("hx", torch.as_tensor(hx, dtype=torch.bool))
        self.register_buffer("hz", torch.as_tensor(hz, dtype=torch.bool))
        self.register_buffer(
            "logicals_x", torch.as_tensor(logicals_x, dtype=torch.bool)
        )
        self.register_buffer(
            "logicals_z", torch.as_tensor(logicals_z, dtype=torch.bool)
        )
        self.syndrome_weight = float(syndrome_weight)
        self.logical_weight = float(logical_weight)
        self.pauli_weight = float(pauli_weight)
        self.deep_supervision_weight = float(deep_supervision_weight)
        self.probability_epsilon = float(probability_epsilon)

    def _components(
        self,
        logits: Tensor,
        syndrome: Tensor,
        pauli: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if logits.ndim != 3 or logits.shape[-1] != 4:
            raise ValueError(f"Expected logits [B,n,4], got {tuple(logits.shape)}.")
        if pauli.shape != logits.shape[:2]:
            raise ValueError(
                f"Expected Pauli labels {tuple(logits.shape[:2])}, "
                f"got {tuple(pauli.shape)}."
            )
        expected_checks = self.hx.shape[0] + self.hz.shape[0]
        if syndrome.shape != (logits.shape[0], expected_checks):
            raise ValueError(
                f"Expected syndrome [B,{expected_checks}], got "
                f"{tuple(syndrome.shape)}."
            )

        probabilities = logits.softmax(dim=-1)
        qx = probabilities[..., 1] + probabilities[..., 2]
        qz = probabilities[..., 2] + probabilities[..., 3]

        # Hx (X stabilizers) detect the Z component; Hz detects X.
        predicted_syndrome = torch.cat(
            (
                _odd_parity_probability(qz, self.hx),
                _odd_parity_probability(qx, self.hz),
            ),
            dim=1,
        )
        eps = self.probability_epsilon
        syndrome_loss = F.binary_cross_entropy(
            predicted_syndrome.clamp(eps, 1.0 - eps),
            syndrome.to(dtype=logits.dtype),
        )

        error_x, error_z = pauli_to_xz(pauli)
        residual_qx = torch.where(error_x, 1.0 - qx, qx)
        residual_qz = torch.where(error_z, 1.0 - qz, qz)

        # A syndrome-free residual is a stabilizer exactly when it also has
        # trivial commutation with a complete logical basis.
        logical_odd = torch.cat(
            (
                _odd_parity_probability(residual_qx, self.logicals_z),
                _odd_parity_probability(residual_qz, self.logicals_x),
            ),
            dim=1,
        )
        logical_loss = -torch.log1p(-logical_odd.clamp(min=0.0, max=1.0 - eps)).mean()

        pauli_loss = F.cross_entropy(
            logits.reshape(-1, 4), pauli.to(dtype=torch.long).reshape(-1)
        )
        total = (
            self.syndrome_weight * syndrome_loss
            + self.logical_weight * logical_loss
            + self.pauli_weight * pauli_loss
        )
        return total, syndrome_loss, logical_loss, pauli_loss

    def forward(
        self,
        logits: Tensor,
        syndrome: Tensor,
        pauli: Tensor,
        iteration_logits: Tensor | None = None,
    ) -> BBLossOutput:
        total, syndrome_loss, logical_loss, pauli_loss = self._components(
            logits, syndrome, pauli
        )
        deep = total.new_zeros(())
        if iteration_logits is not None:
            if iteration_logits.ndim != 4:
                raise ValueError(
                    "iteration_logits must have shape [B,T,n,4], got "
                    f"{tuple(iteration_logits.shape)}."
                )
            if (
                iteration_logits.shape[0] != logits.shape[0]
                or iteration_logits.shape[2:] != logits.shape[1:]
            ):
                raise ValueError(
                    "iteration_logits batch/qubit/Pauli dimensions must match "
                    f"final logits, got {tuple(iteration_logits.shape)} and "
                    f"{tuple(logits.shape)}."
                )
        if (
            iteration_logits is not None
            and self.deep_supervision_weight > 0.0
            and iteration_logits.shape[1] > 1
        ):
            intermediate_terms = []
            # The final state is already supervised above.
            for step in range(iteration_logits.shape[1] - 1):
                intermediate_terms.append(
                    self._components(iteration_logits[:, step], syndrome, pauli)[0]
                )
            if intermediate_terms:
                deep = torch.stack(intermediate_terms).mean()
                total = total + self.deep_supervision_weight * deep

        return BBLossOutput(
            total=total,
            syndrome=syndrome_loss,
            logical=logical_loss,
            pauli=pauli_loss,
            deep_supervision=deep,
        )


__all__ = [
    "BBLossOutput",
    "DegeneracyAwareBPLoss",
    "pauli_to_xz",
]
