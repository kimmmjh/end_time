"""Scoring and OSD post-processing for circuit-level BB decoding.

Belief propagation alone is a weak decoder on a quantum LDPC detector error
model: degenerate fault sets make the Tanner graph's short cycles pathological,
and plain min-sum leaves most shots with an unsatisfied syndrome.  Ordered
statistics decoding repairs this by using the belief-propagation posterior only
as a *reliability ordering* and then solving the syndrome exactly on an
information set.  Every serious BB baseline in the literature is therefore
BP+OSD rather than BP.

:class:`OsdPostprocessor` accepts an arbitrary posterior, so the same
post-processor can be driven by the neural decoder or by vanilla min-sum.  That
keeps a Neural-BP+OSD versus BP+OSD comparison strictly paired: identical
shots, identical post-processor, only the soft input differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


@dataclass(frozen=True)
class BBCircuitOutcomes:
    """Per-shot decoding outcomes on one batch."""

    success: NDArray[np.bool_]
    syndrome_converged: NDArray[np.bool_]

    @property
    def shots(self) -> int:
        return int(self.success.size)

    @property
    def accuracy(self) -> float:
        return float(self.success.mean()) if self.shots else float("nan")

    @property
    def logical_error_rate(self) -> float:
        return 1.0 - self.accuracy

    @property
    def flagged_failure(self) -> float:
        """Failures whose correction does not even reproduce the syndrome."""

        if not self.shots:
            return float("nan")
        return float((~self.syndrome_converged).mean())

    @property
    def unflagged_failure(self) -> float:
        """Failures that satisfy the syndrome but land in the wrong coset."""

        if not self.shots:
            return float("nan")
        return float((~self.success & self.syndrome_converged).mean())


def score_corrections(
    correction: NDArray[np.generic],
    *,
    detectors: NDArray[np.generic],
    observables: NDArray[np.generic],
    check_matrix: Any,
    observable_matrix: Any,
) -> BBCircuitOutcomes:
    """Exactly evaluate hard corrections against detectors and observables."""

    correction = np.asarray(correction, dtype=np.uint8)
    detectors = np.asarray(detectors, dtype=np.uint8)
    observables = np.asarray(observables, dtype=np.uint8)
    check = sp.csr_matrix(check_matrix, dtype=np.uint8)
    observable = sp.csr_matrix(observable_matrix, dtype=np.uint8)

    if correction.ndim != 2 or correction.shape[1] != check.shape[1]:
        raise ValueError(
            f"correction must have shape (shots, {check.shape[1]}), "
            f"got {correction.shape}."
        )
    if detectors.shape != (correction.shape[0], check.shape[0]):
        raise ValueError(
            f"detectors must have shape ({correction.shape[0]}, {check.shape[0]}), "
            f"got {detectors.shape}."
        )
    if observables.shape != (correction.shape[0], observable.shape[0]):
        raise ValueError(
            "observables must have shape "
            f"({correction.shape[0]}, {observable.shape[0]}), got {observables.shape}."
        )

    predicted_syndrome = np.asarray(check @ correction.T, dtype=np.uint8).T % 2
    predicted_observables = (
        np.asarray(observable @ correction.T, dtype=np.uint8).T % 2
    )
    converged = np.all(predicted_syndrome == detectors, axis=1)
    logical_match = np.all(predicted_observables == observables, axis=1)
    # This decoder outputs a physical mechanism correction, not a direct
    # logical-class prediction.  A correction is therefore successful only if
    # it both explains the measured detectors and restores the right logical
    # coset.  Counting a logical coincidence from an unsatisfied syndrome as a
    # success substantially overestimates plain-BP performance.
    success = converged & logical_match
    return BBCircuitOutcomes(success=success, syndrome_converged=converged)


class OsdPostprocessor:
    """Turn a soft posterior into an exact syndrome-satisfying correction.

    Parameters
    ----------
    check_matrix:
        The DEM check matrix, detectors by mechanisms.
    priors:
        Static mechanism probabilities, used when no posterior is supplied.
    method:
        ``OSD_0`` or ``OSD_CS``; ``order`` selects the combination-sweep depth.
    bp_iterations:
        Belief-propagation iterations run inside the post-processor.  With a
        neural posterior supplied this should be small: the point is to hand
        OSD a reliability ordering, not to redo the decoding.
    """

    def __init__(
        self,
        check_matrix: Any,
        *,
        priors: NDArray[np.float64],
        method: str = "OSD_CS",
        order: int = 7,
        bp_iterations: int = 1,
        ms_scaling_factor: float = 0.625,
    ) -> None:
        try:
            from ldpc import BpOsdDecoder
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise ImportError(
                "OSD post-processing requires the 'ldpc' package "
                "(pin ldpc==2.4.1 to match the archived baselines)."
            ) from exc
        if bp_iterations < 1:
            raise ValueError(
                "bp_iterations must be at least one; ldpc reads zero as the "
                "block length rather than as no iterations."
            )

        self._priors = np.asarray(priors, dtype=np.float64)
        self._decoder = BpOsdDecoder(
            sp.csr_matrix(check_matrix, dtype=np.uint8),
            error_channel=list(self._priors),
            max_iter=int(bp_iterations),
            bp_method="ms",
            ms_scaling_factor=float(ms_scaling_factor),
            osd_method=method,
            osd_order=int(order),
        )

    @staticmethod
    def posterior_to_probabilities(
        log_odds: NDArray[np.generic], *, epsilon: float = 1e-12
    ) -> NDArray[np.float64]:
        """Convert posterior LLRs to per-mechanism firing probabilities."""

        values = np.asarray(log_odds, dtype=np.float64)
        # A positive LLR means the mechanism most likely did not fire.
        probabilities = 1.0 / (1.0 + np.exp(np.clip(values, -60.0, 60.0)))
        return np.clip(probabilities, epsilon, 1.0 - epsilon)

    def decode_batch(
        self,
        detectors: NDArray[np.generic],
        *,
        posterior: NDArray[np.generic] | None = None,
    ) -> NDArray[np.uint8]:
        """Decode a batch, optionally reseeded with a per-shot posterior."""

        detector_array = np.asarray(detectors, dtype=np.uint8)
        corrections = np.zeros(
            (detector_array.shape[0], self._priors.size), dtype=np.uint8
        )
        probabilities = (
            None
            if posterior is None
            else self.posterior_to_probabilities(posterior)
        )
        for shot in range(detector_array.shape[0]):
            if probabilities is not None:
                self._decoder.update_channel_probs(probabilities[shot])
            corrections[shot] = np.asarray(
                self._decoder.decode(detector_array[shot]), dtype=np.uint8
            )
        if probabilities is not None:
            # Leave the decoder in its static-prior state for the next caller.
            self._decoder.update_channel_probs(self._priors)
        return corrections


__all__ = ["BBCircuitOutcomes", "OsdPostprocessor", "score_corrections"]
