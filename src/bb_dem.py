"""Detector-error-model Tanner graphs for circuit-level BB decoding.

Code-capacity decoding runs belief propagation on ``Hx``/``Hz`` with one
four-state Pauli belief per data qubit.  Circuit-level decoding cannot: a
fault anywhere in the seven-layer syndrome-extraction cycle propagates through
later CNOTs, so the decoding graph is Stim's detector error model.  Variables
become *fault mechanisms* carrying a single binary "did this fire" belief, and
checks become *detectors*.  The quaternary structure disappears because the DEM
has already factorised each circuit fault into independent mechanisms.

Equivariance
------------
The circuit is generated from the same cyclic shift matrices as the code, so
the group ``Z_ell x Z_m`` still acts on it, and time translation additionally
acts on the bulk rounds.  A mechanism's orbit is therefore the canonical form
of its detector signature under a simultaneous space and time translation.
Because the first frame compares against a noiseless reference cycle and the
last frame is closed noiselessly, mechanisms touching those frames are *not*
time-translation equivalent to bulk mechanisms and are keyed separately.

Only the detector signature enters an orbit key.  Belief propagation messages
depend on the check matrix alone; the observable matrix is used to read off a
logical outcome after decoding, so tying parameters across mechanisms that
share a detector pattern but differ in observables is exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .bb_code import BBCodeSpec

try:
    import stim
except ImportError:  # Keep code-capacity noise usable without Stim.
    stim = None


@dataclass(frozen=True)
class BBDemGraph:
    """A circuit-level BB decoding graph extracted from a Stim DEM."""

    code_name: str
    circuit_schema_version: int
    circuit_noise_model: str
    rounds: int
    detector_frames: int
    ell: int
    m: int
    num_detectors: int
    num_mechanisms: int
    num_observables: int
    check_matrix: sp.csr_matrix
    observable_matrix: sp.csr_matrix
    priors: NDArray[np.float64]
    edge_detector: NDArray[np.int64]
    edge_mechanism: NDArray[np.int64]
    edge_orbit: NDArray[np.int64]
    detector_coordinates: NDArray[np.int64]
    num_orbits: int
    merge_map: NDArray[np.int64]
    dem_fingerprint: str

    def fold_errors(self, errors: NDArray[np.generic]) -> NDArray[np.uint8]:
        """Fold a sampler's per-DEM-mechanism errors into merged columns.

        ``DetectorErrorModel.compile_sampler().sample(..., return_errors=True)``
        reports which of the *original* mechanisms fired.  Duplicate columns are
        merged when the graph is built, so the firing indicators of every
        mechanism sharing a merged column must be XOR-folded together.
        """

        raw = np.asarray(errors, dtype=np.uint8)
        if raw.ndim != 2 or raw.shape[1] != self.merge_map.size:
            raise ValueError(
                f"errors must have shape (shots, {self.merge_map.size}), "
                f"got {raw.shape}."
            )
        folded = np.zeros((raw.shape[0], self.num_mechanisms), dtype=np.uint8)
        np.bitwise_xor.at(folded, (slice(None), self.merge_map), raw)
        return folded

    @property
    def num_edges(self) -> int:
        return int(self.edge_detector.size)

    @property
    def prior_log_odds(self) -> NDArray[np.float64]:
        """``log((1-p)/p)`` per mechanism, the natural BP2 prior LLR.

        Circuit-level priors span orders of magnitude, unlike the constant
        code-capacity prior, so this is a genuinely informative model input.
        """

        clipped = np.clip(self.priors, 1e-15, 1.0 - 1e-15)
        return np.log((1.0 - clipped) / clipped)

    def summary(self) -> str:
        degrees = np.bincount(self.edge_detector, minlength=self.num_detectors)
        return (
            f"{self.code_name} noise={self.circuit_noise_model}, "
            f"noisy_rounds={self.rounds}, "
            f"detector_frames={self.detector_frames}: "
            f"{self.num_detectors} detectors, {self.num_mechanisms} mechanisms, "
            f"{self.num_edges} edges, {self.num_orbits} orbits, "
            f"detector degree {degrees.min()}-{degrees.max()}"
        )


def _require_stim() -> None:
    if stim is None:
        raise ImportError(
            "Circuit-level BB decoding requires Stim. Install the project "
            "dependencies with `pip install -r requirements.txt`."
        )


def _combine_probabilities(first: float, second: float) -> float:
    """Probability that exactly one of two independent mechanisms fires."""

    return first * (1.0 - second) + second * (1.0 - first)


def detector_error_model_fingerprint(dem: "stim.DetectorErrorModel") -> str:
    """Stable identity of a concrete DEM, including all mechanism priors."""

    return hashlib.sha256(str(dem).encode("utf-8")).hexdigest()


def _parse_dem(
    dem: "stim.DetectorErrorModel",
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[float]]:
    detectors: list[tuple[int, ...]] = []
    observables: list[tuple[int, ...]] = []
    priors: list[float] = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detector_ids: list[int] = []
        observable_ids: list[int] = []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detector_ids.append(int(target.val))
            elif target.is_logical_observable_id():
                observable_ids.append(int(target.val))
        if not detector_ids:
            raise ValueError(
                "The detector error model contains a mechanism that flips a "
                "logical observable without firing any detector. The circuit "
                "has an undetectable single-fault logical error."
            )
        detectors.append(tuple(sorted(set(detector_ids))))
        observables.append(tuple(sorted(set(observable_ids))))
        priors.append(float(instruction.args_copy()[0]))

    return detectors, observables, priors


def _merge_duplicates(
    detectors: list[tuple[int, ...]],
    observables: list[tuple[int, ...]],
    priors: list[float],
) -> tuple[
    list[tuple[int, ...]], list[tuple[int, ...]], list[float], NDArray[np.int64]
]:
    """XOR-merge mechanisms with an identical detector/observable signature.

    The returned map sends every original DEM mechanism index to its merged
    column, which is required to fold the sampler's ``return_errors`` output
    into the merged indexing used by the decoder.
    """

    merged: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    out_detectors: list[tuple[int, ...]] = []
    out_observables: list[tuple[int, ...]] = []
    out_priors: list[float] = []
    merge_map: list[int] = []

    for detector_ids, observable_ids, prior in zip(detectors, observables, priors):
        key = (detector_ids, observable_ids)
        index = merged.get(key)
        if index is None:
            index = len(out_priors)
            merged[key] = index
            out_detectors.append(detector_ids)
            out_observables.append(observable_ids)
            out_priors.append(prior)
        else:
            out_priors[index] = _combine_probabilities(out_priors[index], prior)
        merge_map.append(index)

    return (
        out_detectors,
        out_observables,
        out_priors,
        np.asarray(merge_map, dtype=np.int64),
    )


def _canonical_signature(
    detector_ids: Iterable[int],
    coordinates: NDArray[np.int64],
    *,
    ell: int,
    m: int,
    detector_frames: int,
    boundary_width: int,
) -> tuple[Any, tuple[int, ...]]:
    """Return a translation-invariant key and the canonical detector order.

    The key is the lexicographic minimum over every translation that brings one
    of the mechanism's own detectors to spatial cell ``(0, 0)`` and its earliest
    frame to time zero.  Distance to the first and last frame is capped at
    ``boundary_width`` and retained in the key, so bulk mechanisms share an
    orbit while boundary mechanisms do not.
    """

    ids = list(detector_ids)
    rows = coordinates[ids]
    times = rows[:, 3]
    time_min = int(times.min())
    time_max = int(times.max())
    boundary = (
        min(time_min, boundary_width),
        min(detector_frames - 1 - time_max, boundary_width),
    )

    best_key: Any = None
    best_order: tuple[int, ...] = ()
    for reference in range(len(ids)):
        shift_i = int(rows[reference, 1])
        shift_j = int(rows[reference, 2])
        shifted = [
            (
                int(rows[index, 0]),
                int((rows[index, 1] - shift_i) % ell),
                int((rows[index, 2] - shift_j) % m),
                int(rows[index, 3] - time_min),
            )
            for index in range(len(ids))
        ]
        order = sorted(range(len(ids)), key=lambda index: shifted[index])
        candidate = tuple(shifted[index] for index in order)
        if best_key is None or candidate < best_key:
            best_key = candidate
            best_order = tuple(order)

    return (best_key, boundary), best_order


def build_bb_dem_graph(
    code: BBCodeSpec | str,  # noqa: D401
    *,
    rounds: int,
    gate_error_rate: float,
    measurement_error_rate: float | None = None,
    idle_error_rate: float | None = None,
    circuit_noise_model: str = "legacy",
    boundary_width: int = 1,
    merge_duplicates: bool = True,
    circuit: "stim.Circuit | None" = None,
) -> BBDemGraph:
    """Build the circuit-level decoding graph and its equivariant orbit ids.

    ``rounds`` counts noisy extraction cycles.  The circuit has one additional
    perfect closing frame, so the orbit count saturates once
    ``rounds + 1 >= 2 * boundary_width + 2``.  Below that every mechanism
    touches a time boundary and the sharing structure has not reached its bulk
    form.  A memory experiment normally sets ``rounds`` to the code distance.
    """

    _require_stim()
    from .bb_stim_utils import (
        CIRCUIT_SCHEMA_VERSION,
        generate_bb_memory_circuit,
        normalize_bb_circuit_noise_model,
    )

    if isinstance(code, str):
        code = BBCodeSpec.from_name(code)
    resolved_noise_model = normalize_bb_circuit_noise_model(circuit_noise_model)
    if boundary_width < 0:
        raise ValueError("boundary_width must be non-negative.")

    if circuit is None:
        circuit = generate_bb_memory_circuit(
            code,
            rounds=rounds,
            gate_error_rate=gate_error_rate,
            measurement_error_rate=measurement_error_rate,
            idle_error_rate=idle_error_rate,
            circuit_noise_model=resolved_noise_model,
        )
    dem = circuit.detector_error_model(
        decompose_errors=False, allow_gauge_detectors=False
    )

    detectors, observables, priors = _parse_dem(dem)
    if merge_duplicates:
        detectors, observables, priors, merge_map = _merge_duplicates(
            detectors, observables, priors
        )
    else:
        merge_map = np.arange(len(priors), dtype=np.int64)

    num_detectors = int(dem.num_detectors)
    num_mechanisms = len(priors)
    num_observables = int(dem.num_observables)

    detector_frames = rounds + 1
    expected_detectors = detector_frames * code.num_checks
    if num_detectors != expected_detectors:
        raise ValueError(
            "BB circuit detector count does not match its noisy-round "
            f"semantics: expected {expected_detectors} = ({rounds} + 1) * "
            f"{code.num_checks}, got {num_detectors}."
        )

    raw_coordinates = dem.get_detector_coordinates()
    if set(raw_coordinates) != set(range(num_detectors)):
        raise ValueError("Every BB detector must have an explicit coordinate.")
    coordinates = np.zeros((num_detectors, 4), dtype=np.int64)
    for detector_id, values in raw_coordinates.items():
        if len(values) != 4:
            raise ValueError(
                "Every BB detector must carry (check_type, i, j, time) "
                f"coordinates, got {values!r} for detector {detector_id}."
            )
        coordinates[int(detector_id)] = [int(round(value)) for value in values]
    times, time_counts = np.unique(coordinates[:, 3], return_counts=True)
    if not np.array_equal(times, np.arange(detector_frames)) or not np.all(
        time_counts == code.num_checks
    ):
        raise ValueError(
            "BB detector coordinates do not contain exactly one complete check "
            f"frame at every time 0..{detector_frames - 1}."
        )

    edge_detector: list[int] = []
    edge_mechanism: list[int] = []
    edge_orbit: list[int] = []
    orbit_ids: dict[Any, int] = {}

    for mechanism, detector_ids in enumerate(detectors):
        key, order = _canonical_signature(
            detector_ids,
            coordinates,
            ell=code.ell,
            m=code.m,
            detector_frames=detector_frames,
            boundary_width=boundary_width,
        )
        for position, index in enumerate(order):
            edge_key = (key, position)
            orbit = orbit_ids.get(edge_key)
            if orbit is None:
                orbit = len(orbit_ids)
                orbit_ids[edge_key] = orbit
            edge_detector.append(int(detector_ids[index]))
            edge_mechanism.append(mechanism)
            edge_orbit.append(orbit)

    detector_array = np.asarray(edge_detector, dtype=np.int64)
    mechanism_array = np.asarray(edge_mechanism, dtype=np.int64)
    orbit_array = np.asarray(edge_orbit, dtype=np.int64)
    # Row-major over detectors keeps the segmented check update contiguous.
    order = np.lexsort((mechanism_array, detector_array))
    detector_array = detector_array[order]
    mechanism_array = mechanism_array[order]
    orbit_array = orbit_array[order]

    ones = np.ones(detector_array.size, dtype=np.uint8)
    check_matrix = sp.csr_matrix(
        (ones, (detector_array, mechanism_array)),
        shape=(num_detectors, num_mechanisms),
    )
    observable_rows = [
        observable
        for observable_ids in observables
        for observable in observable_ids
    ]
    observable_cols = [
        mechanism
        for mechanism, observable_ids in enumerate(observables)
        for _ in observable_ids
    ]
    observable_matrix = sp.csr_matrix(
        (
            np.ones(len(observable_rows), dtype=np.uint8),
            (observable_rows, observable_cols),
        ),
        shape=(num_observables, num_mechanisms),
    )

    return BBDemGraph(
        code_name=code.name,
        circuit_schema_version=CIRCUIT_SCHEMA_VERSION,
        circuit_noise_model=resolved_noise_model,
        rounds=rounds,
        detector_frames=detector_frames,
        ell=code.ell,
        m=code.m,
        num_detectors=num_detectors,
        num_mechanisms=num_mechanisms,
        num_observables=num_observables,
        check_matrix=check_matrix,
        observable_matrix=observable_matrix,
        priors=np.asarray(priors, dtype=np.float64),
        edge_detector=detector_array,
        edge_mechanism=mechanism_array,
        edge_orbit=orbit_array,
        detector_coordinates=coordinates,
        num_orbits=len(orbit_ids),
        merge_map=merge_map,
        dem_fingerprint=detector_error_model_fingerprint(dem),
    )


__all__ = [
    "BBDemGraph",
    "build_bb_dem_graph",
    "detector_error_model_fingerprint",
]
