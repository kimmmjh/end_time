"""Local feed-forward DEM CNN followed by global residual decoding.

The trainable network is BBDemCNN: variable-degree neighborhoods are reduced
by orbit-aware graph convolutions, without padding, pooling or recurrence.
``forward`` returns mechanism LLRs for the existing circuit loss. Hard local
decisions and the classical backend live in the inference-only ``decode`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from ._bb_dem_cnn import BBDemCNN


class ResidualDecoder(Protocol):
    """Backend over the original DEM columns; probabilities refer to e XOR a."""

    def decode_batch(self, syndrome: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Return binary residual corrections [batch, num_mechanisms]."""
        ...


@dataclass(frozen=True)
class LocalDEMResult:
    log_odds: Tensor
    local_correction: Tensor
    residual_detectors: Tensor
    residual_log_odds: Tensor


@dataclass(frozen=True)
class HierarchicalDEMResult:
    local: LocalDEMResult
    residual_correction: Tensor
    correction: Tensor
    global_used: Tensor
    syndrome_converged: Tensor


class BBHierarchicalDemCNN(BBDemCNN):
    """Confidence-gated local correction plus global syndrome completion.

    ``forward(detectors)`` has exactly the BBDemCNN training contract and state
    dict layout, so existing CNN weights can initialize this model. ``depth``
    controls the local graph receptive field (default 2), not hierarchy levels.

    A mechanism is locally flipped only if P(fault) >= ``confidence_threshold``
    and, by default, it touches an active detector. Every other mechanism is
    left unchanged, not frozen: the global decoder can change *any* column.
    The threshold is an inference hyperparameter, not a calibrated guarantee.

    ``decode`` skips the backend on zero residual syndromes and otherwise uses
    a supplied decode_batch backend, or lazily constructs the existing dense
    CPU DirectOSD0 reference. Backend failures propagate; inconsistent outputs
    raise instead of silently being accepted as decoded corrections.

    Graph buffers remain nonpersistent, as in BBDemCNN. Reconstruct with the
    same fingerprint-checked graph on checkpoint loading. Save the confidence
    threshold and candidate policy separately as experiment configuration.
    """

    def __init__(self, graph, *, width=64, depth=2, edge_chunk_size=4096,
                 gradient_checkpoint=True, confidence_threshold=0.9,
                 active_candidates_only=True):
        if not math.isfinite(confidence_threshold) or not 0.5 < confidence_threshold < 1:
            raise ValueError("confidence_threshold must be finite and between 0.5 and 1.")
        super().__init__(graph, width=width, depth=depth,
                         edge_chunk_size=edge_chunk_size,
                         gradient_checkpoint=gradient_checkpoint)
        self.confidence_threshold = float(confidence_threshold)
        self.active_candidates_only = bool(active_candidates_only)
        self._edge_chunk_size = edge_chunk_size
        # Keep sparse until the optional reference backend is actually needed.
        self._check_matrix = graph.check_matrix.copy()
        self._default_backend = None

    def _syndrome(self, correction: Tensor) -> Tensor:
        """Exact sparse H e over GF(2), retaining the original hyperedges."""
        parity = torch.zeros((correction.shape[0], self.num_detectors),
                             dtype=torch.long, device=correction.device)
        for start in range(0, self.edge_detector.numel(), self._edge_chunk_size):
            stop = start + self._edge_chunk_size
            parity.index_add_(1, self.edge_detector[start:stop],
                              correction[:, self.edge_mechanism[start:stop]].long())
        return parity.remainder_(2).to(torch.uint8)

    @torch.no_grad()
    def preprocess(self, detectors: Tensor) -> LocalDEMResult:
        """Propose a local correction a and form s XOR H a and residual LLRs.

        All proposals use the same input syndrome and are applied together.
        This method is inference-only; use forward for differentiable training.
        """
        log_odds = self(detectors)
        if not torch.isfinite(log_odds).all():
            raise ValueError("Local network returned non-finite mechanism LLRs.")
        syndrome = detectors.to(device=log_odds.device, dtype=torch.uint8)
        # LLR = log(P(no fault)/P(fault)); confident faults have negative LLR.
        cutoff = math.log1p(-self.confidence_threshold) - math.log(self.confidence_threshold)
        selected = log_odds <= cutoff
        if self.active_candidates_only:
            counts = torch.zeros_like(log_odds, dtype=torch.long)
            for start in range(0, self.edge_detector.numel(), self._edge_chunk_size):
                stop = start + self._edge_chunk_size
                counts.index_add_(1, self.edge_mechanism[start:stop],
                                  syndrome[:, self.edge_detector[start:stop]].long())
            selected = selected & (counts > 0)
        local = selected.to(torch.uint8)
        residual = syndrome ^ self._syndrome(local)
        residual_llrs = log_odds * (1 - 2 * local.to(log_odds.dtype))
        return LocalDEMResult(log_odds, local, residual, residual_llrs)

    @torch.no_grad()
    def decode(self, detectors: Tensor, *, backend: ResidualDecoder | None = None
               ) -> HierarchicalDEMResult:
        """Return a XOR b, where H b = s XOR H a, on the model's device.

        Predicted observable flips are O @ result.correction (mod 2). True
        observables and sampled mechanism labels never enter this method.
        Neural residual probabilities are heuristic backend inputs, not
        independent channel priors. No differentiable backend is assumed.
        """
        local = self.preprocess(detectors)
        used = local.residual_detectors.bool().any(dim=1)
        residual_correction = torch.zeros_like(local.local_correction)
        if used.any():
            if backend is None:
                if self._default_backend is None:
                    from src._direct_osd import DirectOSD0
                    self._default_backend = DirectOSD0(self._check_matrix)
                backend = self._default_backend
            syndrome_np = local.residual_detectors[used].cpu().numpy()
            # Convert before sigmoid to avoid avoidable float32 saturation.
            probabilities = torch.sigmoid(
                -local.residual_log_odds[used].double()).cpu().numpy()
            decoded = np.asarray(backend.decode_batch(syndrome_np, probabilities))
            if decoded.shape != probabilities.shape or not np.isin(decoded, [0, 1]).all():
                raise ValueError("Backend must return binary residual corrections [batch, mechanisms].")
            residual_correction[used] = torch.as_tensor(
                decoded, dtype=torch.uint8, device=residual_correction.device)
        correction = local.local_correction ^ residual_correction
        syndrome = detectors.to(device=correction.device, dtype=torch.uint8)
        converged = (self._syndrome(correction) == syndrome).all(dim=1)
        if not converged.all():
            raise RuntimeError("Residual decoder returned a syndrome-inconsistent correction.")
        return HierarchicalDEMResult(local, residual_correction, correction, used, converged)


__all__ = ["BBHierarchicalDemCNN", "LocalDEMResult", "HierarchicalDEMResult", "ResidualDecoder"]
