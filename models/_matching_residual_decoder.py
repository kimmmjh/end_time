"""Combine a translation-invariant neural residual with circuit-level MWPM."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pymatching
import torch
from torch import Tensor, nn
from torch.nn import functional as F


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

    supports_paired_baseline = True
    recommendation_name = "hybrid"

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
        self._last_residual_logits: Tensor | None = None

        # A freshly trained neural residual is not trusted by default.  These
        # buffers are part of state_dict(), so a calibrated deployment decision
        # survives checkpointing and resume.
        self.register_buffer("gate_enabled", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("gate_margin", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "gate_temperature", torch.tensor(1.0, dtype=torch.float32)
        )

    @property
    def last_baseline_classes(self) -> Tensor:
        """Return PyMatching classes cached by the most recent forward pass."""

        if self._last_matching_classes is None:
            raise RuntimeError(
                "last_baseline_classes requires a preceding forward() call."
            )
        return self._last_matching_classes

    @property
    def last_residual_logits(self) -> Tensor:
        """Return raw neural logits cached by the most recent forward pass."""

        if self._last_residual_logits is None:
            raise RuntimeError(
                "last_residual_logits requires a preceding forward() call."
            )
        return self._last_residual_logits

    def configure_gate(
        self,
        *,
        enabled: bool,
        temperature: float = 1.0,
        margin: float = 0.0,
    ) -> None:
        """Set the residual gate using validated, checkpointed values."""

        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("margin must be finite and non-negative.")
        with torch.no_grad():
            self.gate_enabled.fill_(enabled)
            self.gate_temperature.fill_(temperature)
            self.gate_margin.fill_(margin)

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
        self._last_residual_logits = residual_logits

        if not bool(self.gate_enabled.item()):
            gated_residual_classes = torch.zeros(
                residual_logits.shape[0],
                dtype=torch.long,
                device=residual_logits.device,
            )
        else:
            temperature = self.gate_temperature.to(
                device=residual_logits.device, dtype=residual_logits.dtype
            )
            margin = self.gate_margin.to(
                device=residual_logits.device, dtype=residual_logits.dtype
            )
            scaled = residual_logits / temperature
            best_nonzero_scores, best_nonzero_indices = scaled[:, 1:].max(dim=1)
            apply_residual = (best_nonzero_scores - scaled[:, 0]) > margin
            gated_residual_classes = torch.where(
                apply_residual,
                best_nonzero_indices + 1,
                torch.zeros_like(best_nonzero_indices),
            )

        # A hard one-hot decision avoids tie-breaking after the XOR permutation:
        # equality with the margin must always fall back to residual class zero,
        # regardless of the numerical value of the absolute MWPM class.
        decision_logits = F.one_hot(
            gated_residual_classes, num_classes=self.num_classes
        ).to(dtype=residual_logits.dtype)

        return self._xor_permute(decision_logits, matching_classes)

    def loss_inputs(
        self,
        final_logits: Tensor,
        true_classes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert final-class tensors to residual-class loss inputs.

        This should be called immediately after :meth:`forward`.  Training uses
        the raw, uncalibrated neural logits; the gate affects deployment and
        metrics only.  Class zero explicitly means that MWPM succeeded.
        """

        matching_classes = self.last_baseline_classes
        residual_logits = self.last_residual_logits
        if true_classes.ndim != 1 or true_classes.shape[0] != final_logits.shape[0]:
            raise ValueError(
                "Expected true classes with shape (B,), got "
                f"{tuple(true_classes.shape)}."
            )
        if residual_logits.shape != final_logits.shape:
            raise ValueError(
                "Cached residual logits do not match final logits: "
                f"{tuple(residual_logits.shape)} != {tuple(final_logits.shape)}."
            )

        matching_classes = matching_classes.to(device=residual_logits.device)
        residual_classes = torch.bitwise_xor(
            true_classes.to(device=residual_logits.device, dtype=torch.long),
            matching_classes,
        )
        return residual_logits, residual_classes

    @staticmethod
    def _fit_temperature(logits: Tensor, targets: Tensor) -> tuple[float, float]:
        """Fit one scalar temperature with a deterministic NLL grid search."""

        # A bounded scalar search is robust for small/imbalanced calibration
        # sets and avoids introducing optimizer state into model calibration.
        log_temperatures = torch.linspace(
            math.log(0.05),
            math.log(20.0),
            steps=81,
            dtype=logits.dtype,
            device=logits.device,
        )
        best_temperature = 1.0
        best_nll = math.inf
        best_distance_from_one = math.inf
        with torch.no_grad():
            for log_temperature in log_temperatures:
                temperature = log_temperature.exp()
                nll = float(F.cross_entropy(logits / temperature, targets).item())
                distance_from_one = abs(float(log_temperature.item()))
                if nll < best_nll - 1e-12 or (
                    abs(nll - best_nll) <= 1e-12
                    and distance_from_one < best_distance_from_one
                ):
                    best_temperature = float(temperature.item())
                    best_nll = nll
                    best_distance_from_one = distance_from_one
        return best_temperature, best_nll

    def calibrate_gate(
        self,
        residual_logits: Tensor,
        true_classes: Tensor,
        matching_classes: Tensor,
        confidence_z: float = 1.96,
    ) -> dict[str, bool | int | float]:
        """Calibrate temperature and a conservative MWPM-residual gate.

        Candidate margins are evaluated by their *paired* change relative to
        MWPM on the same shots.  A correction is a rescue when MWPM was wrong
        and the selected residual is exact; it is a harm when MWPM was already
        correct.  The disabled, pure-MWPM decision with zero net gain is always
        a candidate.  The gate is enabled only if the best candidate has a
        strictly positive lower confidence bound on net gain.
        """

        if not math.isfinite(confidence_z) or confidence_z < 0.0:
            raise ValueError("confidence_z must be finite and non-negative.")
        if residual_logits.ndim != 2 or residual_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected residual logits (B, {self.num_classes}), got "
                f"{tuple(residual_logits.shape)}."
            )
        sample_count = residual_logits.shape[0]
        if sample_count < 1:
            raise ValueError("Calibration requires at least one sample.")
        if true_classes.shape != (sample_count,) or matching_classes.shape != (
            sample_count,
        ):
            raise ValueError(
                "Expected true_classes and matching_classes with shape (B,), "
                f"got {tuple(true_classes.shape)} and "
                f"{tuple(matching_classes.shape)}."
            )

        logits = residual_logits.detach().to(device="cpu", dtype=torch.float32)
        truth = true_classes.detach().to(device="cpu", dtype=torch.long)
        baseline = matching_classes.detach().to(device="cpu", dtype=torch.long)
        if torch.any((truth < 0) | (truth >= self.num_classes)):
            raise ValueError("true_classes contains an out-of-range class.")
        if torch.any((baseline < 0) | (baseline >= self.num_classes)):
            raise ValueError("matching_classes contains an out-of-range class.")

        residual_targets = torch.bitwise_xor(truth, baseline)
        temperature, residual_nll = self._fit_temperature(
            logits, residual_targets
        )
        scaled = logits / temperature
        nonzero_scores, nonzero_indices = scaled[:, 1:].max(dim=1)
        nonzero_classes = nonzero_indices + 1
        advantages = nonzero_scores - scaled[:, 0]

        target_numpy = residual_targets.numpy()
        predicted_numpy = nonzero_classes.numpy()
        advantages_numpy = advantages.numpy()
        eligible = np.flatnonzero(
            np.isfinite(advantages_numpy) & (advantages_numpy > 0.0)
        )

        baseline_correct = int(np.count_nonzero(target_numpy == 0))
        baseline_accuracy = baseline_correct / sample_count
        best: dict[str, bool | int | float] = {
            "enabled": False,
            "temperature": temperature,
            "margin": 0.0,
            "calibration_samples": sample_count,
            "baseline_accuracy": baseline_accuracy,
            "hybrid_accuracy": baseline_accuracy,
            "corrections": 0,
            "correction_rate": 0.0,
            "rescues": 0,
            "rescue_rate": 0.0,
            "harms": 0,
            "harm_rate": 0.0,
            "neutral_changes": 0,
            "net_gain": 0,
            "net_gain_rate": 0.0,
            "net_gain_standard_error": 0.0,
            "net_gain_lcb": 0.0,
            "confidence_z": float(confidence_z),
            "residual_nll": residual_nll,
        }

        if eligible.size:
            order = eligible[
                np.argsort(-advantages_numpy[eligible], kind="stable")
            ]
            sorted_advantages = advantages_numpy[order]
            is_rescue = (
                (target_numpy[order] != 0)
                & (predicted_numpy[order] == target_numpy[order])
            ).astype(np.int64)
            is_harm = (target_numpy[order] == 0).astype(np.int64)
            cumulative_rescues = np.cumsum(is_rescue)
            cumulative_harms = np.cumsum(is_harm)

            # Evaluate only complete equal-score groups so the stored threshold
            # reproduces the calibration decision exactly.
            group_ends = np.flatnonzero(
                np.r_[
                    sorted_advantages[1:] != sorted_advantages[:-1],
                    True,
                ]
            )
            best_lcb = 0.0  # Pure MWPM is the conservative initial candidate.
            best_margin = math.inf
            for end in group_ends:
                corrections = int(end + 1)
                rescues = int(cumulative_rescues[end])
                harms = int(cumulative_harms[end])
                neutral_changes = corrections - rescues - harms
                net_gain = rescues - harms
                net_gain_rate = net_gain / sample_count
                # Each paired shot contributes X in {-1, 0, +1}: harm,
                # unchanged, or rescue.  Use the unbiased sample variance of X
                # (Bessel correction), then divide by N for the standard error
                # of its mean.  With one shot there is no variance estimate, so
                # treating the uncertainty as infinite conservatively prevents
                # calibration from enabling the gate.
                if sample_count > 1:
                    sum_squared_deviations = (
                        rescues
                        + harms
                        - sample_count * net_gain_rate**2
                    )
                    sample_variance = max(sum_squared_deviations, 0.0) / (
                        sample_count - 1
                    )
                    standard_error = math.sqrt(sample_variance / sample_count)
                else:
                    standard_error = math.inf
                lcb = (
                    net_gain_rate - confidence_z * standard_error
                    if sample_count > 1
                    else -math.inf
                )

                current_score = float(sorted_advantages[end])
                if end + 1 < sorted_advantages.size:
                    next_score = float(sorted_advantages[end + 1])
                    margin = (current_score + next_score) / 2.0
                else:
                    margin = 0.0

                better_lcb = lcb > best_lcb + 1e-12
                conservative_tie = (
                    abs(lcb - best_lcb) <= 1e-12 and margin > best_margin
                )
                if better_lcb or conservative_tie:
                    hybrid_accuracy = baseline_accuracy + net_gain_rate
                    best_lcb = lcb
                    best_margin = margin
                    best = {
                        "enabled": True,
                        "temperature": temperature,
                        "margin": margin,
                        "calibration_samples": sample_count,
                        "baseline_accuracy": baseline_accuracy,
                        "hybrid_accuracy": hybrid_accuracy,
                        "corrections": corrections,
                        "correction_rate": corrections / sample_count,
                        "rescues": rescues,
                        "rescue_rate": rescues / sample_count,
                        "harms": harms,
                        "harm_rate": harms / sample_count,
                        "neutral_changes": neutral_changes,
                        "net_gain": net_gain,
                        "net_gain_rate": net_gain_rate,
                        "net_gain_standard_error": standard_error,
                        "net_gain_lcb": lcb,
                        "confidence_z": float(confidence_z),
                        "residual_nll": residual_nll,
                    }

        enabled = bool(best["enabled"]) and float(best["net_gain_lcb"]) > 0.0
        if not enabled:
            # This also handles numerical/tie cases by returning to the exact
            # pure-MWPM candidate.
            best["enabled"] = False
            best["margin"] = 0.0
        self.configure_gate(
            enabled=enabled,
            temperature=float(best["temperature"]),
            margin=float(best["margin"]),
        )
        return best
