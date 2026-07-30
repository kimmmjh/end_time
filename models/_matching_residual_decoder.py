"""Combine a translation-invariant neural residual with circuit-level MWPM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymatching
import torch
from torch import Tensor, nn


class MatchingResidualDecoder(nn.Module):
    """Correct the logical prediction of a static PyMatching decoder.

    The neural decoder predicts the residual homology class between the true
    error and the correction selected by PyMatching.  If ``m`` is PyMatching's
    logical class and ``r`` is the residual class, the final logical class is
    ``m XOR r``.

    Args:
        residual_decoder: Module mapping ``(B, 2, T, L^2)`` detector tensors to
            residual logits of shape ``(B, 2**num_observables)``.
        matching: An already constructed :class:`pymatching.Matching`.
        circuit: Alternatively, a Stim circuit from which a decomposed detector
            error model and matching graph will be built.
        detector_error_model: Alternatively, a Stim detector error model.
        num_observables: Number of logical-observable bits.  It is four for the
            2D toric code used by this repository.
        enable_correlations: Build and run PyMatching's correlated two-pass
            decoder.  A supplied ``matching`` must also have been constructed
            with correlations enabled when this is true.

    Exactly one of ``matching``, ``circuit``, or ``detector_error_model`` must
    be supplied.  PyMatching runs on CPU and is deliberately outside the
    autograd graph; gradients flow through ``residual_decoder``.
    """

    def __init__(
        self,
        residual_decoder: nn.Module,
        matching: pymatching.Matching | None = None,
        *,
        circuit: Any | None = None,
        detector_error_model: Any | None = None,
        num_observables: int = 4,
        enable_correlations: bool = True,
    ) -> None:
        super().__init__()
        if num_observables < 1:
            raise ValueError("num_observables must be positive.")

        sources = sum(
            source is not None
            for source in (matching, circuit, detector_error_model)
        )
        if sources != 1:
            raise ValueError(
                "Provide exactly one of matching, circuit, or "
                "detector_error_model."
            )

        if circuit is not None:
            detector_error_model = circuit.detector_error_model(
                decompose_errors=True
            )
        if matching is None:
            matching = pymatching.Matching.from_detector_error_model(
                detector_error_model,
                enable_correlations=enable_correlations,
            )

        self.residual_decoder = residual_decoder
        self.matching = matching
        self.num_observables = num_observables
        self.num_classes = 1 << num_observables
        self.enable_correlations = enable_correlations
        self._last_matching_classes: Tensor | None = None

    @property
    def last_baseline_classes(self) -> Tensor:
        """Return PyMatching classes cached by the most recent forward pass."""

        if self._last_matching_classes is None:
            raise RuntimeError(
                "last_baseline_classes requires a preceding forward() call."
            )
        return self._last_matching_classes

    @staticmethod
    def flatten_detectors(syndrome: Tensor | np.ndarray) -> np.ndarray:
        """Convert model detector order to Stim/PyMatching detector order.

        The data generators expose ``(B, sector, time, L^2)``, whereas Stim
        numbers detectors as ``(time, sector, position)``.
        """

        if isinstance(syndrome, Tensor):
            detectors = syndrome.detach().to(device="cpu").numpy()
        else:
            detectors = np.asarray(syndrome)

        if detectors.ndim == 2:
            flat = detectors
        elif detectors.ndim == 4:
            if detectors.shape[1] != 2:
                raise ValueError(
                    "Expected two detector sectors in shape "
                    f"(B, 2, T, L^2), got {detectors.shape}."
                )
            flat = detectors.transpose(0, 2, 1, 3).reshape(
                detectors.shape[0], -1
            )
        else:
            raise ValueError(
                "Expected flattened detectors (B, D) or model detector shape "
                f"(B, 2, T, L^2), got {detectors.shape}."
            )

        return np.ascontiguousarray(flat, dtype=np.uint8)

    def matching_observables(
        self,
        syndrome: Tensor | np.ndarray,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Decode detector samples and return padded observable bits ``(B, O)``."""

        flat = self.flatten_detectors(syndrome)
        decoded = np.asarray(
            self.matching.decode_batch(
                flat,
                enable_correlations=self.enable_correlations,
            ),
            dtype=np.uint8,
        )

        batch = flat.shape[0]
        if decoded.ndim == 1:
            if batch == 1:
                decoded = decoded.reshape(1, -1)
            elif self.num_observables == 1 and decoded.shape[0] == batch:
                decoded = decoded.reshape(batch, 1)
            else:
                raise RuntimeError(
                    "PyMatching returned an unexpected prediction shape "
                    f"{decoded.shape} for batch size {batch}."
                )
        if decoded.ndim != 2 or decoded.shape[0] != batch:
            raise RuntimeError(
                "PyMatching returned an unexpected prediction shape "
                f"{decoded.shape} for batch size {batch}."
            )
        if decoded.shape[1] > self.num_observables:
            raise RuntimeError(
                "PyMatching returned more fault observables than expected: "
                f"{decoded.shape[1]} > {self.num_observables}."
            )
        if decoded.shape[1] < self.num_observables:
            decoded = np.pad(
                decoded,
                ((0, 0), (0, self.num_observables - decoded.shape[1])),
            )

        if device is None and isinstance(syndrome, Tensor):
            device = syndrome.device
        return torch.as_tensor(decoded, dtype=torch.long, device=device)

    def matching_classes(
        self,
        syndrome: Tensor | np.ndarray,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return big-endian integer classes for PyMatching observables."""

        observables = self.matching_observables(syndrome, device=device)
        powers = 1 << torch.arange(
            self.num_observables - 1,
            -1,
            -1,
            dtype=torch.long,
            device=observables.device,
        )
        return (observables * powers).sum(dim=1)

    def _xor_permute(self, logits: Tensor, matching_classes: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                "Expected residual/final logits with shape "
                f"(B, {self.num_classes}), got {tuple(logits.shape)}."
            )
        if matching_classes.ndim != 1 or matching_classes.shape[0] != logits.shape[0]:
            raise ValueError(
                "Expected one matching class per logit row, got "
                f"{tuple(matching_classes.shape)} for {tuple(logits.shape)}."
            )

        class_indices = torch.arange(
            self.num_classes, device=logits.device, dtype=torch.long
        )
        permutation = torch.bitwise_xor(
            class_indices.unsqueeze(0),
            matching_classes.to(device=logits.device, dtype=torch.long).unsqueeze(1),
        )
        return logits.gather(1, permutation)

    def forward(self, syndrome: Tensor) -> Tensor:
        """Return logits over the final absolute logical class."""

        residual_logits = self.residual_decoder(syndrome)
        matching_classes = self.matching_classes(
            syndrome, device=residual_logits.device
        )
        self._last_matching_classes = matching_classes
        return self._xor_permute(residual_logits, matching_classes)

    def loss_inputs(
        self,
        final_logits: Tensor,
        true_classes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert final-class tensors to residual-class loss inputs.

        This should be called immediately after :meth:`forward`.  The XOR
        permutation is its own inverse, so it recovers residual logits without
        detaching them from autograd.  Residual targets are useful for class
        weighting because class zero explicitly means that MWPM succeeded.
        """

        matching_classes = self.last_baseline_classes
        if true_classes.ndim != 1 or true_classes.shape[0] != final_logits.shape[0]:
            raise ValueError(
                "Expected true classes with shape (B,), got "
                f"{tuple(true_classes.shape)}."
            )

        matching_classes = matching_classes.to(device=final_logits.device)
        residual_logits = self._xor_permute(final_logits, matching_classes)
        residual_classes = torch.bitwise_xor(
            true_classes.to(device=final_logits.device, dtype=torch.long),
            matching_classes,
        )
        return residual_logits, residual_classes
