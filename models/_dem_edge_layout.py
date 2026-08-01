"""Sparse Stim-DEM edge layout used by neural-weighted matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, csr_matrix


def _edge_key(node1: int, node2: int | None) -> tuple[int, int]:
    """Return one stable key for an undirected edge or boundary half-edge."""

    if node2 is None:
        return int(node1), -1
    first, second = sorted((int(node1), int(node2)))
    return first, second


def _fault_mask(fault_ids: set[int], num_observables: int) -> np.ndarray:
    mask = np.zeros(num_observables, dtype=np.uint8)
    for fault_id in fault_ids:
        if not 0 <= int(fault_id) < num_observables:
            raise ValueError(
                "Matching edge contains an out-of-range logical fault id: "
                f"{fault_id} for {num_observables} observables."
            )
        mask[int(fault_id)] = 1
    return mask


@dataclass(frozen=True)
class DemEdgeArrays:
    """NumPy/SciPy representation of a merged graphlike DEM."""

    endpoints: np.ndarray
    base_weights: np.ndarray
    base_logits: np.ndarray
    error_probabilities: np.ndarray
    faults: csc_matrix
    check_matrix: csc_matrix
    mechanism_to_edge: csr_matrix
    geometry: np.ndarray


class DemEdgeLayout:
    """Align Stim fault mechanisms with the sparse PyMatching graph.

    Stim's decomposed detector error model can contain correlated components
    separated by ``^`` and several independent mechanisms can collapse onto the
    same PyMatching edge.  This class splits every mechanism into graphlike
    components and stores a sparse GF(2) mechanism-to-merged-edge map.  As a
    result, sampled mechanism bits can be converted into the parity target for
    every matching edge without treating correlated components as independent.
    """

    def __init__(
        self,
        detector_error_model: Any,
        *,
        lattice_size: int,
        rounds: int,
        num_observables: int,
    ) -> None:
        if lattice_size < 1:
            raise ValueError("lattice_size must be positive.")
        if rounds < 1:
            raise ValueError("rounds must be positive.")
        if num_observables < 1:
            raise ValueError("num_observables must be positive.")

        self.detector_error_model = detector_error_model.flattened()
        self.lattice_size = int(lattice_size)
        self.rounds = int(rounds)
        self.num_observables = int(num_observables)
        self.num_detectors = int(self.detector_error_model.num_detectors)
        expected_detectors = self.rounds * 2 * self.lattice_size**2
        if self.num_detectors != expected_detectors:
            raise ValueError(
                "The DEM detector count does not match (rounds, sectors, L^2): "
                f"{self.num_detectors} != {expected_detectors}."
            )

        self.matching = pymatching.Matching.from_detector_error_model(
            self.detector_error_model,
            enable_correlations=False,
        )
        self.arrays = self._build_arrays()
        self.num_edges = int(self.arrays.endpoints.shape[0])
        self.num_error_mechanisms = int(self.arrays.mechanism_to_edge.shape[0])

    def _build_arrays(self) -> DemEdgeArrays:
        matching_edges = self.matching.edges()
        if not matching_edges:
            raise ValueError("The detector error model produced no matching edges.")

        endpoints = np.empty((len(matching_edges), 2), dtype=np.int64)
        base_weights = np.empty(len(matching_edges), dtype=np.float64)
        error_probabilities = np.empty(len(matching_edges), dtype=np.float64)
        fault_columns = np.zeros(
            (self.num_observables, len(matching_edges)), dtype=np.uint8
        )
        key_to_edge: dict[tuple[int, int], int] = {}

        check_rows: list[int] = []
        check_columns: list[int] = []
        for edge_index, (node1, node2, data) in enumerate(matching_edges):
            key = _edge_key(node1, node2)
            if key in key_to_edge:
                raise ValueError(f"PyMatching returned a duplicate merged edge {key}.")
            key_to_edge[key] = edge_index
            endpoints[edge_index] = key

            weight = float(data["weight"])
            if not np.isfinite(weight):
                raise ValueError(f"Non-finite base matching weight on edge {key}.")
            base_weights[edge_index] = weight

            probability = float(data.get("error_probability", -1.0))
            if not 0.0 <= probability <= 1.0:
                # The DEM matcher normally records this value.  Recover it from
                # the exact log-likelihood-ratio weight for defensive support of
                # older PyMatching versions.
                probability = 1.0 / (1.0 + np.exp(np.clip(weight, -60.0, 60.0)))
            error_probabilities[edge_index] = probability

            fault_columns[:, edge_index] = _fault_mask(
                set(data.get("fault_ids", set())), self.num_observables
            )
            check_rows.append(key[0])
            check_columns.append(edge_index)
            if key[1] >= 0:
                check_rows.append(key[1])
                check_columns.append(edge_index)

        check_matrix = csc_matrix(
            (
                np.ones(len(check_rows), dtype=np.uint8),
                (check_rows, check_columns),
            ),
            shape=(self.num_detectors, len(matching_edges)),
            dtype=np.uint8,
        )
        faults = csc_matrix(fault_columns, dtype=np.uint8)

        toggled_pairs: set[tuple[int, int]] = set()
        mechanism_index = 0
        for instruction in self.detector_error_model:
            if instruction.type != "error":
                continue
            component: list[Any] = []
            components: list[list[Any]] = []
            for target in instruction.targets_copy():
                if target.is_separator():
                    components.append(component)
                    component = []
                else:
                    component.append(target)
            components.append(component)

            for targets in components:
                detectors = [
                    int(target.val)
                    for target in targets
                    if target.is_relative_detector_id()
                ]
                logical_ids = {
                    int(target.val)
                    for target in targets
                    if target.is_logical_observable_id()
                }
                if len(detectors) > 2:
                    raise ValueError(
                        "The decomposed DEM is not graphlike: error mechanism "
                        f"{mechanism_index} has a component touching "
                        f"{len(detectors)} detectors."
                    )
                if not detectors:
                    if logical_ids:
                        raise ValueError(
                            "The DEM contains a detector-free logical mechanism; "
                            "no syndrome-conditioned matching decoder can learn "
                            f"mechanism {mechanism_index}."
                        )
                    continue
                key = _edge_key(
                    detectors[0], detectors[1] if len(detectors) == 2 else None
                )
                edge_index = key_to_edge.get(key)
                if edge_index is None:
                    raise ValueError(
                        "A graphlike DEM component is missing from PyMatching's "
                        f"merged graph: mechanism={mechanism_index}, edge={key}."
                    )
                expected_faults = {
                    int(index)
                    for index in np.flatnonzero(fault_columns[:, edge_index])
                }
                if logical_ids != expected_faults:
                    raise ValueError(
                        "Parallel DEM components with the same endpoints have "
                        "incompatible logical fault masks, so a single sparse "
                        "matching edge is ambiguous: "
                        f"edge={key}, component={sorted(logical_ids)}, "
                        f"matching={sorted(expected_faults)}."
                    )
                pair = (mechanism_index, edge_index)
                # Two identical components in one mechanism cancel over GF(2).
                if pair in toggled_pairs:
                    toggled_pairs.remove(pair)
                else:
                    toggled_pairs.add(pair)
            mechanism_index += 1

        expected_mechanisms = int(self.detector_error_model.num_errors)
        if mechanism_index != expected_mechanisms:
            raise ValueError(
                "DEM error-instruction count mismatch: "
                f"parsed {mechanism_index}, expected {expected_mechanisms}."
            )
        incidence_rows = [pair[0] for pair in toggled_pairs]
        incidence_columns = [pair[1] for pair in toggled_pairs]
        mechanism_to_edge = csr_matrix(
            (
                np.ones(len(toggled_pairs), dtype=np.uint8),
                (incidence_rows, incidence_columns),
            ),
            shape=(mechanism_index, len(matching_edges)),
            dtype=np.uint8,
        )

        geometry = self._edge_geometry(endpoints, base_weights)
        # A matching edge weight is log((1-p)/p), hence -weight is its prior
        # logit.  Using the weight directly also exactly respects independent
        # parallel-edge merging performed by PyMatching.
        base_logits = -base_weights
        return DemEdgeArrays(
            endpoints=endpoints,
            base_weights=base_weights,
            base_logits=base_logits,
            error_probabilities=error_probabilities,
            faults=faults,
            check_matrix=check_matrix,
            mechanism_to_edge=mechanism_to_edge,
            geometry=geometry,
        )

    def _edge_geometry(
        self, endpoints: np.ndarray, base_weights: np.ndarray
    ) -> np.ndarray:
        """Return translation-safe relative features for every sparse edge."""

        lattice_area = self.lattice_size**2
        time_denominator = max(self.rounds - 1, 1)
        spatial_denominator = max(self.lattice_size // 2, 1)
        geometry = np.zeros((endpoints.shape[0], 8), dtype=np.float32)

        for edge_index, (node1, node2) in enumerate(endpoints):
            time1, remainder1 = divmod(int(node1), 2 * lattice_area)
            sector1, position1 = divmod(remainder1, lattice_area)
            x1, y1 = divmod(position1, self.lattice_size)

            boundary = node2 < 0
            if boundary:
                dx = dy = dt = 0.0
                pair_type = 3
            else:
                time2, remainder2 = divmod(int(node2), 2 * lattice_area)
                sector2, position2 = divmod(remainder2, lattice_area)
                x2, y2 = divmod(position2, self.lattice_size)
                raw_dx = abs(x1 - x2)
                raw_dy = abs(y1 - y2)
                dx = min(raw_dx, self.lattice_size - raw_dx) / spatial_denominator
                dy = min(raw_dy, self.lattice_size - raw_dy) / spatial_denominator
                dt = abs(time1 - time2) / time_denominator
                pair_type = sector1 + sector2  # 0, 1, or 2; order independent.

            geometry[edge_index, 0:3] = (dx, dy, dt)
            geometry[edge_index, 3 + pair_type] = 1.0
            geometry[edge_index, 7] = np.tanh(base_weights[edge_index] / 10.0)

        return geometry

    def edge_targets(self, fired_mechanisms: np.ndarray) -> np.ndarray:
        """Convert fired DEM instructions into merged-edge parity labels."""

        fired = np.asarray(fired_mechanisms, dtype=np.uint8)
        if fired.ndim != 2 or fired.shape[1] != self.num_error_mechanisms:
            raise ValueError(
                "Expected fired DEM mechanisms with shape "
                f"(batch, {self.num_error_mechanisms}), got {fired.shape}."
            )
        products = csr_matrix(fired) @ self.arrays.mechanism_to_edge
        targets = products.toarray().astype(np.uint8, copy=False)
        targets &= 1
        return targets
