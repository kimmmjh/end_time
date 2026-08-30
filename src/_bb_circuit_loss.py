"""Degeneracy-aware objective for the circuit-level BB neural BP decoder.

The circuit-level decoder returns one binary belief per detector-error-model
fault mechanism.  A sampled fault set is not a unique decoding target: any two
mechanism sets with the same detector signature and the same observable
signature are equally good corrections.  The primary terms below therefore ask
the correction to reproduce the measured detectors and to leave a residual with
trivial observable parity, exactly as the code-capacity objective does for
stabilizers and logical operators.  Direct mechanism cross entropy is kept only
as a small optimisation aid.

All parity marginals are computed in the log-likelihood-ratio domain.  A BB
detector is incident on up to roughly two hundred and fifty mechanisms, so the
naive product of ``1 - 2q`` factors is both slow and numerically fragile;
accumulating ``log|tanh(lambda/2)|`` with a separate sign parity is stable and
uses the same segmented scatter primitives as the decoder itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class BBCircuitLossOutput:
    """Individual loss terms, including the differentiable total."""

    total: Tensor
    syndrome: Tensor
    logical: Tensor
    mechanism: Tensor
    deep_supervision: Tensor


def _sparse_edges(matrix: Any) -> tuple[Tensor, Tensor, int]:
    """Return row-major ``(row, column)`` index tensors of a binary matrix."""

    sparse = sp.csr_matrix(matrix)
    coo = sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    rows = torch.as_tensor(np.asarray(coo.row)[order], dtype=torch.long)
    columns = torch.as_tensor(np.asarray(coo.col)[order], dtype=torch.long)
    return rows, columns, int(sparse.shape[0])


class CircuitDegeneracyAwareLoss(nn.Module):
    """Detector/observable objective with a mechanism cross-entropy aid."""

    def __init__(
        self,
        *,
        check_matrix: Any,
        observable_matrix: Any,
        syndrome_weight: float = 1.0,
        logical_weight: float = 1.0,
        mechanism_weight: float = 0.1,
        deep_supervision_weight: float = 0.2,
        probability_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        for name, value in (
            ("syndrome_weight", syndrome_weight),
            ("logical_weight", logical_weight),
            ("mechanism_weight", mechanism_weight),
            ("deep_supervision_weight", deep_supervision_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if not 0.0 < probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie in (0, 0.5).")

        check_rows, check_columns, num_checks = _sparse_edges(check_matrix)
        observable_rows, observable_columns, num_observables = _sparse_edges(
            observable_matrix
        )
        self.register_buffer("check_rows", check_rows)
        self.register_buffer("check_columns", check_columns)
        self.register_buffer("observable_rows", observable_rows)
        self.register_buffer("observable_columns", observable_columns)
        self.num_checks = num_checks
        self.num_observables = num_observables

        self.syndrome_weight = float(syndrome_weight)
        self.logical_weight = float(logical_weight)
        self.mechanism_weight = float(mechanism_weight)
        self.deep_supervision_weight = float(deep_supervision_weight)
        self.probability_epsilon = float(probability_epsilon)

    def _odd_parity_probability(
        self, log_odds: Tensor, rows: Tensor, columns: Tensor, num_rows: int
    ) -> Tensor:
        """Exact odd-parity marginal of each row under factorised beliefs."""

        eps = self.probability_epsilon
        factors = torch.tanh(0.5 * log_odds)[:, columns]
        factors = factors.clamp(-1.0 + eps, 1.0 - eps)

        batch_size = log_odds.shape[0]
        magnitude = torch.zeros(
            (batch_size, num_rows), device=log_odds.device, dtype=log_odds.dtype
        # tanh(0) is exactly zero.  log(abs(tanh(0))) has a finite forward
        # limit after the product below, but its backward pass is NaN.  Clamp
        # the magnitude itself so an initially uncertain LLR remains fully
        # differentiable.
        ).index_add(1, rows, factors.abs().clamp_min(eps).log())
        negatives = torch.zeros(
            (batch_size, num_rows), device=log_odds.device, dtype=log_odds.dtype
        ).index_add(1, rows, (factors < 0).to(log_odds.dtype))
        sign = 1.0 - 2.0 * torch.remainder(negatives, 2.0)
        return (0.5 * (1.0 - sign * magnitude.exp())).clamp(eps, 1.0 - eps)

    def _components(
        self, log_odds: Tensor, detectors: Tensor, mechanisms: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if log_odds.ndim != 2:
            raise ValueError(
                f"Expected posterior log odds [B,N], got {tuple(log_odds.shape)}."
            )
        if detectors.shape != (log_odds.shape[0], self.num_checks):
            raise ValueError(
                f"Expected detectors [B,{self.num_checks}], got "
                f"{tuple(detectors.shape)}."
            )

        predicted = self._odd_parity_probability(
            log_odds, self.check_rows, self.check_columns, self.num_checks
        )
        syndrome_loss = F.binary_cross_entropy(
            predicted, detectors.to(dtype=log_odds.dtype)
        )

        if mechanisms is None:
            zero = log_odds.new_zeros(())
            return syndrome_loss * self.syndrome_weight, syndrome_loss, zero, zero

        if mechanisms.shape != log_odds.shape:
            raise ValueError(
                f"Expected mechanism labels {tuple(log_odds.shape)}, got "
                f"{tuple(mechanisms.shape)}."
            )
        truth = mechanisms.to(dtype=log_odds.dtype)
        # Residual log odds: flipping a truly fired mechanism negates its LLR,
        # so the residual is trivial exactly when every observable parity is even.
        residual_log_odds = log_odds * (1.0 - 2.0 * truth)
        logical_odd = self._odd_parity_probability(
            residual_log_odds,
            self.observable_rows,
            self.observable_columns,
            self.num_observables,
        )
        logical_loss = -torch.log1p(-logical_odd).mean()

        # A positive LLR means "did not fire", hence the negation.
        mechanism_loss = F.binary_cross_entropy_with_logits(-log_odds, truth)

        total = (
            self.syndrome_weight * syndrome_loss
            + self.logical_weight * logical_loss
            + self.mechanism_weight * mechanism_loss
        )
        return total, syndrome_loss, logical_loss, mechanism_loss

    def forward(
        self,
        log_odds: Tensor,
        detectors: Tensor,
        mechanisms: Tensor | None = None,
        iteration_log_odds: Tensor | None = None,
    ) -> BBCircuitLossOutput:
        total, syndrome_loss, logical_loss, mechanism_loss = self._components(
            log_odds, detectors, mechanisms
        )
        deep = total.new_zeros(())

        if iteration_log_odds is not None:
            if iteration_log_odds.ndim != 3:
                raise ValueError(
                    "iteration_log_odds must have shape [B,T,N], got "
                    f"{tuple(iteration_log_odds.shape)}."
                )
            if (
                iteration_log_odds.shape[0] != log_odds.shape[0]
                or iteration_log_odds.shape[2] != log_odds.shape[1]
            ):
                raise ValueError(
                    "iteration_log_odds batch/mechanism dimensions must match "
                    f"the final posterior, got {tuple(iteration_log_odds.shape)} "
                    f"and {tuple(log_odds.shape)}."
                )
            if self.deep_supervision_weight > 0.0 and iteration_log_odds.shape[1] > 1:
                intermediate = [
                    # The final state is already supervised above.
                    self._components(
                        iteration_log_odds[:, step], detectors, mechanisms
                    )[0]
                    for step in range(iteration_log_odds.shape[1] - 1)
                ]
                if intermediate:
                    deep = torch.stack(intermediate).mean()
                    total = total + self.deep_supervision_weight * deep

        return BBCircuitLossOutput(
            total=total,
            syndrome=syndrome_loss,
            logical=logical_loss,
            mechanism=mechanism_loss,
            deep_supervision=deep,
        )


__all__ = ["BBCircuitLossOutput", "CircuitDegeneracyAwareLoss"]
