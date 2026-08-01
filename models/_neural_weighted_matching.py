"""Circuit decoder using ConvGRU-conditioned, shot-specific MWPM weights."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pymatching
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._dem_edge_layout import DemEdgeLayout


class NeuralWeightedMatchingDecoder(nn.Module):
    """Predict sparse DEM edge probabilities, then run MWPM with those weights.

    Training uses fired Stim DEM mechanisms as an edge-parity supervision proxy;
    MWPM is intentionally skipped in training because it is non-differentiable.
    Evaluation rebuilds a standard PyMatching graph for each shot from a fixed
    check matrix and the conditional weights ``-logit(q_e | syndrome)``.
    """

    supports_paired_baseline = True
    requires_batch_metadata = True
    recommendation_name = "neural_weighted_mwpm"

    def __init__(
        self,
        edge_network: nn.Module,
        *,
        detector_error_model: Any,
        lattice_size: int,
        rounds: int,
        num_observables: int = 4,
        weight_clip: float = 20.0,
    ) -> None:
        super().__init__()
        if weight_clip <= 0.0:
            raise ValueError("weight_clip must be positive.")
        self.edge_network = edge_network
        self.layout = DemEdgeLayout(
            detector_error_model,
            lattice_size=lattice_size,
            rounds=rounds,
            num_observables=num_observables,
        )
        self.matching = self.layout.matching
        self.num_observables = int(num_observables)
        self.num_classes = 1 << self.num_observables
        self.weight_clip = float(weight_clip)

        arrays = self.layout.arrays
        self.register_buffer(
            "edge_endpoints",
            torch.as_tensor(arrays.endpoints, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "edge_geometry",
            torch.as_tensor(arrays.geometry, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "base_edge_logits",
            torch.as_tensor(arrays.base_logits, dtype=torch.float32),
            persistent=False,
        )
        self._last_baseline_classes: Tensor | None = None
        self._last_edge_logits: Tensor | None = None

    @property
    def last_baseline_classes(self) -> Tensor:
        if self._last_baseline_classes is None:
            raise RuntimeError(
                "last_baseline_classes requires an evaluation forward() call."
            )
        return self._last_baseline_classes

    @property
    def last_edge_logits(self) -> Tensor:
        if self._last_edge_logits is None:
            raise RuntimeError("last_edge_logits requires a preceding forward() call.")
        return self._last_edge_logits

    @staticmethod
    def flatten_detectors(syndrome: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(syndrome, Tensor):
            detectors = syndrome.detach().to(device="cpu").numpy()
        else:
            detectors = np.asarray(syndrome)
        if detectors.ndim == 2:
            flat = detectors
        elif detectors.ndim == 4:
            if detectors.shape[1] != 2:
                raise ValueError(
                    "Expected two detector sectors in (B,2,T,L^2), got "
                    f"{detectors.shape}."
                )
            flat = detectors.transpose(0, 2, 1, 3).reshape(detectors.shape[0], -1)
        else:
            raise ValueError(
                "Expected flattened (B,D) or model (B,2,T,L^2) detectors, got "
                f"{detectors.shape}."
            )
        return np.ascontiguousarray(flat, dtype=np.uint8)

    def _observable_bits_to_classes(
        self,
        observable_bits: np.ndarray,
        *,
        expected_batch: int,
        device: torch.device,
    ) -> Tensor:
        decoded = np.asarray(observable_bits, dtype=np.uint8)
        if decoded.ndim == 1:
            if expected_batch == 1:
                decoded = decoded.reshape(1, -1)
            elif self.num_observables == 1 and decoded.shape[0] == expected_batch:
                decoded = decoded.reshape(expected_batch, 1)
            else:
                raise RuntimeError(
                    f"Unexpected PyMatching output shape {decoded.shape}."
                )
        if decoded.ndim != 2 or decoded.shape[0] != expected_batch:
            raise RuntimeError(f"Unexpected PyMatching output shape {decoded.shape}.")
        if decoded.shape[1] > self.num_observables:
            raise RuntimeError(
                "PyMatching returned more observables than expected: "
                f"{decoded.shape[1]} > {self.num_observables}."
            )
        if decoded.shape[1] < self.num_observables:
            decoded = np.pad(
                decoded,
                ((0, 0), (0, self.num_observables - decoded.shape[1])),
            )
        powers = 1 << np.arange(self.num_observables - 1, -1, -1, dtype=np.int64)
        classes = decoded.astype(np.int64) @ powers
        return torch.as_tensor(classes, dtype=torch.long, device=device)

    def _baseline_classes(self, flat: np.ndarray, *, device: torch.device) -> Tensor:
        decoded = self.matching.decode_batch(flat, enable_correlations=False)
        return self._observable_bits_to_classes(
            np.asarray(decoded), expected_batch=flat.shape[0], device=device
        )

    def _dynamic_classes(
        self,
        flat: np.ndarray,
        edge_logits: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        # For a sparse physical-error graph the MWPM cost is the log odds
        # log((1-q)/q), exactly -logit(q).  Negative weights are supported by
        # Matching.from_check_matrix; clipping only prevents numerical extremes.
        weights = (
            -edge_logits.detach().float().clamp(
                min=-self.weight_clip, max=self.weight_clip
            )
        ).to(device="cpu").numpy()
        predictions = np.zeros((flat.shape[0], self.num_observables), dtype=np.uint8)
        arrays = self.layout.arrays
        for shot_index, shot_weights in enumerate(weights):
            matching = pymatching.Matching.from_check_matrix(
                arrays.check_matrix,
                weights=np.asarray(shot_weights, dtype=np.float64),
                faults_matrix=arrays.faults,
                use_virtual_boundary_node=True,
            )
            decoded = np.asarray(matching.decode(flat[shot_index]), dtype=np.uint8)
            predictions[shot_index, : min(decoded.size, self.num_observables)] = (
                decoded[: self.num_observables]
            )
        return self._observable_bits_to_classes(
            predictions, expected_batch=flat.shape[0], device=device
        )

    def forward(self, syndrome: Tensor) -> Tensor:
        edge_logits = self.edge_network(
            syndrome,
            self.edge_endpoints,
            self.edge_geometry,
            self.base_edge_logits,
        )
        self._last_edge_logits = edge_logits

        if self.training:
            # Trainer uses loss_inputs() below.  Avoid millions of unnecessary
            # CPU MWPM reconstructions while gradients are being accumulated.
            self._last_baseline_classes = None
            return edge_logits.new_zeros((syndrome.shape[0], self.num_classes))

        flat = self.flatten_detectors(syndrome)
        baseline_classes = self._baseline_classes(flat, device=edge_logits.device)
        dynamic_classes = self._dynamic_classes(
            flat, edge_logits, device=edge_logits.device
        )
        self._last_baseline_classes = baseline_classes
        return F.one_hot(dynamic_classes, num_classes=self.num_classes).to(
            dtype=edge_logits.dtype
        )

    def loss_inputs(
        self,
        final_logits: Tensor,
        true_classes: Tensor,
        *,
        batch_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return conditional edge logits and DEM edge-parity targets."""

        del final_logits, true_classes
        if batch_metadata is None or "dem_error_mechanisms" not in batch_metadata:
            raise ValueError(
                "Neural weighted MWPM training requires dem_error_mechanisms "
                "from CircuitLevelDataGenerator.generate_batch_with_metadata()."
            )
        targets = self.layout.edge_targets(
            np.asarray(batch_metadata["dem_error_mechanisms"])
        )
        logits = self.last_edge_logits
        target_tensor = torch.as_tensor(
            targets, dtype=logits.dtype, device=logits.device
        )
        if target_tensor.shape != logits.shape:
            raise RuntimeError(
                "DEM edge target/logit mismatch: "
                f"{tuple(target_tensor.shape)} != {tuple(logits.shape)}."
            )
        return logits.reshape(-1), target_tensor.reshape(-1)
