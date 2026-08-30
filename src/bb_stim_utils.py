"""Stim circuits for circuit-level bivariate-bicycle (BB) code data generation.

The toric generator in :mod:`src.stim_utils` assumes nearest-neighbour
weight-four stabilizers on a square lattice and cannot express a BB code.  BB
checks have weight six and connect cells that are far apart in any planar
embedding, so this module builds its own syndrome-extraction cycle directly
from the ``BBCodeSpec`` polynomial terms.

Schedule
--------
One cycle interleaves the X- and Z-check CNOTs over seven layers.  A layer
assignment is chosen per Tanner-edge *orbit* rather than per edge, so the
schedule is automatically invariant under the cyclic translation group that
defines the code.  Two families of constraint make a schedule legal:

``rainbow``
    Each X ancilla, each Z ancilla, each left-block data qubit and each
    right-block data qubit touches exactly six CNOTs, so those six layer
    indices must be distinct.

``determinism``
    An X ancilla acts on a data qubit while a Z ancilla reads it.  If the
    X-CNOT precedes the Z-CNOT, the Z ancilla inherits the X ancilla's
    superposition.  For every pair of checks the number of shared qubits
    where this happens must be even, otherwise the detector is random even in
    a noiseless circuit.  Because the nine displacements ``a_k + b_j`` are
    pairwise distinct for both published BB codes, each pair of checks shares
    exactly one left-block and one right-block qubit, and the condition
    reduces to the per-orbit predicate

    ``[t(X,L,a_k) < t(Z,L,b_j)] == [t(X,R,b_j) < t(Z,R,a_k)]``

    for every ``k`` and ``j``.  Depth six admits no solution; depth seven does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import stim
except ImportError:  # Keep code-capacity noise usable without Stim.
    stim = None

from .bb_code import BBCodeSpec

# Layer index for each of the twelve Tanner-edge orbits.  Verified below by
# ``validate_schedule`` and, end to end, by Stim's own determinism check.
DEFAULT_SCHEDULE: dict[str, tuple[int, int, int]] = {
    "x_left": (0, 1, 5),
    "x_right": (2, 3, 4),
    "z_left": (2, 3, 4),
    "z_right": (5, 6, 0),
}
SCHEDULE_DEPTH = 7
# Increment this whenever the public circuit semantics change.  Version 2 makes
# ``rounds`` mean noisy extraction cycles and adds a separate perfect closing
# frame; version-1 checkpoints therefore must not be mixed with this graph.
CIRCUIT_SCHEMA_VERSION = 2


def _require_stim() -> None:
    if stim is None:
        raise ImportError(
            "Circuit-level BB noise requires Stim. Install the project "
            "dependencies with `pip install -r requirements.txt`."
        )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def validate_schedule(schedule: dict[str, tuple[int, int, int]]) -> int:
    """Check the rainbow and determinism conditions; return the depth."""

    required = ("x_left", "x_right", "z_left", "z_right")
    missing = [key for key in required if key not in schedule]
    if missing:
        raise ValueError(f"Schedule is missing {missing}.")
    for key in required:
        if len(schedule[key]) != 3:
            raise ValueError(f"Schedule entry {key!r} must hold three layers.")

    xl = tuple(int(value) for value in schedule["x_left"])
    xr = tuple(int(value) for value in schedule["x_right"])
    zl = tuple(int(value) for value in schedule["z_left"])
    zr = tuple(int(value) for value in schedule["z_right"])
    if min(xl + xr + zl + zr) < 0:
        raise ValueError("Schedule layers must be non-negative.")

    for name, group in (
        ("X ancilla", xl + xr),
        ("Z ancilla", zl + zr),
        ("left data qubit", xl + zl),
        ("right data qubit", xr + zr),
    ):
        if len(set(group)) != 6:
            raise ValueError(
                f"Schedule assigns a repeated layer to one {name}: {group}."
            )

    for k in range(3):
        for j in range(3):
            if (xl[k] < zl[j]) != (xr[j] < zr[k]):
                raise ValueError(
                    "Schedule violates detector determinism at "
                    f"(a_{k}, b_{j}); such a circuit has random detectors "
                    "even without noise."
                )
    return max(xl + xr + zl + zr) + 1


def search_schedules(
    depth: int, *, limit: int | None = None
) -> list[dict[str, tuple[int, int, int]]]:
    """Enumerate every legal orbit schedule of the given depth."""

    import itertools

    slots = range(depth)
    found: list[dict[str, tuple[int, int, int]]] = []
    for xl in itertools.permutations(slots, 3):
        free_after_xl = [slot for slot in slots if slot not in xl]
        for zl in itertools.permutations(free_after_xl, 3):
            for xr in itertools.permutations(free_after_xl, 3):
                if len(set(xl + xr)) != 6:
                    continue
                for zr in itertools.permutations(
                    [slot for slot in slots if slot not in xr], 3
                ):
                    candidate = {
                        "x_left": xl,
                        "x_right": xr,
                        "z_left": zl,
                        "z_right": zr,
                    }
                    try:
                        validate_schedule(candidate)
                    except ValueError:
                        continue
                    found.append(candidate)
                    if limit is not None and len(found) >= limit:
                        return found
    return found


@dataclass(frozen=True)
class BBCircuitLayout:
    """Qubit indexing and detector ordering for one BB memory circuit."""

    code: BBCodeSpec
    rounds: int
    depth: int

    @property
    def cells(self) -> int:
        return self.code.cells

    @property
    def left_base(self) -> int:
        return 0

    @property
    def right_base(self) -> int:
        return self.cells

    @property
    def x_ancilla_base(self) -> int:
        return 2 * self.cells

    @property
    def z_ancilla_base(self) -> int:
        return 3 * self.cells

    @property
    def num_qubits(self) -> int:
        return 4 * self.cells

    @property
    def num_ancillas(self) -> int:
        return 2 * self.cells

    @property
    def detector_frames(self) -> int:
        """Noisy measurement frames plus the perfect closing boundary."""

        return self.rounds + 1

    @property
    def num_detectors(self) -> int:
        return self.detector_frames * self.num_ancillas

    @property
    def num_observables(self) -> int:
        return 2 * self.code.k


def _shift_cell(code: BBCodeSpec, cell: int, dx: int, dy: int) -> int:
    i, j = divmod(cell, code.m)
    return ((i + dx) % code.ell) * code.m + ((j + dy) % code.m)


def _cnot_layers(
    code: BBCodeSpec,
    schedule: dict[str, tuple[int, int, int]],
    layout: BBCircuitLayout,
) -> list[list[tuple[int, int]]]:
    """Return per-layer ``(control, target)`` CNOT pairs for one cycle."""

    a_terms = ((3, 0), (0, 1), (0, 2))
    b_terms = ((0, 3), (1, 0), (2, 0))
    layers: list[list[tuple[int, int]]] = [[] for _ in range(layout.depth)]
    cells = layout.cells

    # X ancillas are controls: they *apply* X to the data qubits they check.
    for term_index, (dx, dy) in enumerate(a_terms):
        layer = layers[schedule["x_left"][term_index]]
        for cell in range(cells):
            layer.append(
                (
                    layout.x_ancilla_base + cell,
                    layout.left_base + _shift_cell(code, cell, dx, dy),
                )
            )
    for term_index, (dx, dy) in enumerate(b_terms):
        layer = layers[schedule["x_right"][term_index]]
        for cell in range(cells):
            layer.append(
                (
                    layout.x_ancilla_base + cell,
                    layout.right_base + _shift_cell(code, cell, dx, dy),
                )
            )

    # Z ancillas are targets: the data qubits copy their Z parity onto them.
    # Transposed blocks give the negated displacements of Hz = [B^T | A^T].
    for term_index, (dx, dy) in enumerate(b_terms):
        layer = layers[schedule["z_left"][term_index]]
        for cell in range(cells):
            layer.append(
                (
                    layout.left_base + _shift_cell(code, cell, -dx, -dy),
                    layout.z_ancilla_base + cell,
                )
            )
    for term_index, (dx, dy) in enumerate(a_terms):
        layer = layers[schedule["z_right"][term_index]]
        for cell in range(cells):
            layer.append(
                (
                    layout.right_base + _shift_cell(code, cell, -dx, -dy),
                    layout.z_ancilla_base + cell,
                )
            )

    for index, layer in enumerate(layers):
        touched = [qubit for pair in layer for qubit in pair]
        if len(touched) != len(set(touched)):
            raise ValueError(f"CNOT layer {index} contains a qubit collision.")
    return layers


def _check_support_from_layers(
    layers: Sequence[Sequence[tuple[int, int]]], layout: BBCircuitLayout
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Recover Hx and Hz from the scheduled CNOTs for an independent check."""

    cells = layout.cells
    hx = np.zeros((cells, 2 * cells), dtype=np.uint8)
    hz = np.zeros((cells, 2 * cells), dtype=np.uint8)
    for layer in layers:
        for control, target in layer:
            if control >= layout.x_ancilla_base and control < layout.z_ancilla_base:
                hx[control - layout.x_ancilla_base, target] ^= 1
            elif target >= layout.z_ancilla_base:
                hz[target - layout.z_ancilla_base, control] ^= 1
            else:  # pragma: no cover - defensive.
                raise ValueError("Unexpected CNOT orientation in the schedule.")
    return hx, hz


def _append_cycle(
    circuit: "stim.Circuit",
    *,
    layout: BBCircuitLayout,
    layers: Sequence[Sequence[tuple[int, int]]],
    gate_error_rate: float,
    measurement_error_rate: float,
    idle_error_rate: float,
) -> None:
    """Append one ancilla-based BB syndrome-extraction cycle."""

    cells = layout.cells
    x_ancillas = [layout.x_ancilla_base + cell for cell in range(cells)]
    z_ancillas = [layout.z_ancilla_base + cell for cell in range(cells)]
    ancillas = x_ancillas + z_ancillas
    data = list(range(2 * cells))
    noisy = gate_error_rate > 0.0

    circuit.append("R", ancillas)
    if noisy:
        circuit.append("X_ERROR", ancillas, gate_error_rate)
    circuit.append("H", x_ancillas)
    if noisy:
        circuit.append("DEPOLARIZE1", x_ancillas, gate_error_rate)

    for layer in layers:
        targets = [qubit for pair in layer for qubit in pair]
        circuit.append("CX", targets)
        if noisy:
            circuit.append("DEPOLARIZE2", targets, gate_error_rate)
        if idle_error_rate > 0.0:
            busy = set(targets)
            idle = [qubit for qubit in data if qubit not in busy]
            if idle:
                circuit.append("DEPOLARIZE1", idle, idle_error_rate)

    circuit.append("H", x_ancillas)
    if noisy:
        circuit.append("DEPOLARIZE1", x_ancillas, gate_error_rate)
    if measurement_error_rate > 0.0:
        circuit.append("X_ERROR", ancillas, measurement_error_rate)
    circuit.append("M", ancillas)


def _append_logical_sheets(
    circuit: "stim.Circuit", code: BBCodeSpec
) -> None:
    """Mark Pauli-frame correlation sheets in ``[logical X, logical Z]`` order."""

    for index, row in enumerate(code.logicals_x):
        targets = [stim.target_x(int(q)) for q in np.flatnonzero(row)]
        circuit.append("OBSERVABLE_INCLUDE", targets, index)
    for index, row in enumerate(code.logicals_z):
        targets = [stim.target_z(int(q)) for q in np.flatnonzero(row)]
        circuit.append("OBSERVABLE_INCLUDE", targets, code.k + index)


def _append_detector_frame(
    circuit: "stim.Circuit",
    *,
    layout: BBCircuitLayout,
    time_index: int,
) -> None:
    """Compare the two most recent ancilla measurements check by check."""

    num_ancillas = layout.num_ancillas
    for check_type in (0, 1):
        for cell in range(layout.cells):
            offset = check_type * layout.cells + cell
            i, j = divmod(cell, layout.code.m)
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-num_ancillas + offset),
                    stim.target_rec(-2 * num_ancillas + offset),
                ],
                [float(check_type), float(i), float(j), float(time_index)],
            )


def generate_bb_memory_circuit(
    code: BBCodeSpec | str,
    *,
    rounds: int,
    gate_error_rate: float,
    measurement_error_rate: float,
    idle_error_rate: float = 0.0,
    schedule: dict[str, tuple[int, int, int]] | None = None,
    check_schedule: bool = True,
) -> "stim.Circuit":
    """Build a noisy BB memory circuit with detector annotations.

    Detectors are ordered ``(time, check_type, cell)`` with all X checks before
    all Z checks in every frame, matching the code-capacity syndrome layout
    ``[Hx z | Hz x]``.  Observables are correlation sheets in
    ``[logicals_x, logicals_z]`` order, so X, Y and Z logical components of one
    shot stay correlated.

    ``rounds`` is the number of noisy syndrome-extraction cycles.  A noiseless
    reference cycle precedes them so that every detector is deterministic, and
    a separate noiseless closing cycle follows them so that faults late in the
    final noisy cycle cannot become undetectable logical errors.  Consequently
    the circuit returns ``rounds + 1`` detector frames.

    Gate noise follows the toric convention in :mod:`src.stim_utils`: a bit
    flip after reset, one-qubit depolarizing after each Hadamard, two-qubit
    depolarizing after each CNOT layer and a readout flip before measurement.
    ``idle_error_rate`` additionally depolarizes data qubits that sit out a
    CNOT layer; it defaults to zero to match the toric circuits, but a BB cycle
    is seven layers deep, so a non-zero value is the more realistic setting.

    Set ``check_schedule=False`` only to explore alternative CNOT orderings.
    The layer-collision check still runs, but the determinism condition is
    skipped, so the caller must verify the result with
    :func:`assert_detectors_deterministic` before trusting any decoding result.
    """

    _require_stim()
    if isinstance(code, str):
        code = BBCodeSpec.from_name(code)
    _validate_probability("gate_error_rate", gate_error_rate)
    _validate_probability("measurement_error_rate", measurement_error_rate)
    _validate_probability("idle_error_rate", idle_error_rate)
    if gate_error_rate > 0.75:
        raise ValueError(
            "gate_error_rate must be at most 0.75 because the BB circuit uses "
            "DEPOLARIZE1 after Hadamards."
        )
    if idle_error_rate > 0.75:
        raise ValueError(
            "idle_error_rate must be at most 0.75 for a one-qubit depolarizing "
            "channel."
        )
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}.")

    chosen = dict(DEFAULT_SCHEDULE if schedule is None else schedule)
    if check_schedule:
        depth = validate_schedule(chosen)
    else:
        depth = max(
            value
            for key in ("x_left", "x_right", "z_left", "z_right")
            for value in chosen[key]
        ) + 1
    layout = BBCircuitLayout(code=code, rounds=rounds, depth=depth)
    layers = _cnot_layers(code, chosen, layout)

    scheduled_hx, scheduled_hz = _check_support_from_layers(layers, layout)
    if not np.array_equal(scheduled_hx, code.hx):
        raise ValueError("Scheduled X-check support does not reproduce code.hx.")
    if not np.array_equal(scheduled_hz, code.hz):
        raise ValueError("Scheduled Z-check support does not reproduce code.hz.")

    circuit = stim.Circuit()
    circuit.append("R", range(2 * layout.cells))

    # Project the reset product state into a stabilizer eigenspace so that the
    # very first noisy frame already has deterministic detectors.
    _append_cycle(
        circuit,
        layout=layout,
        layers=layers,
        gate_error_rate=0.0,
        measurement_error_rate=0.0,
        idle_error_rate=0.0,
    )
    _append_logical_sheets(circuit, code)

    for time_index in range(rounds):
        _append_cycle(
            circuit,
            layout=layout,
            layers=layers,
            gate_error_rate=gate_error_rate,
            measurement_error_rate=measurement_error_rate,
            idle_error_rate=idle_error_rate,
        )
        _append_detector_frame(circuit, layout=layout, time_index=time_index)

    # This is a boundary measurement, not one of the requested noisy rounds.
    # Keeping it separate preserves every final-cycle hook/data fault in the
    # detector error model.
    _append_cycle(
        circuit,
        layout=layout,
        layers=layers,
        gate_error_rate=0.0,
        measurement_error_rate=0.0,
        idle_error_rate=0.0,
    )
    _append_detector_frame(circuit, layout=layout, time_index=rounds)

    _append_logical_sheets(circuit, code)
    return circuit


def assert_detectors_deterministic(circuit: "stim.Circuit") -> None:
    """Raise if any detector is random in the noiseless circuit."""

    _require_stim()
    noiseless = circuit.without_noise()
    try:
        # Stim's stabilizer-flow analysis is deterministic.  Sampling one
        # noiseless shot can miss a random detector with probability 1/2 and is
        # therefore not a validity check.
        noiseless.detector_error_model(
            decompose_errors=False, allow_gauge_detectors=False
        )
    except ValueError as exc:
        raise ValueError(
            "The noiseless circuit has a non-deterministic detector or logical "
            "correlation sheet; the CNOT schedule is not a valid QND memory "
            "circuit."
        ) from exc


__all__ = [
    "BBCircuitLayout",
    "CIRCUIT_SCHEMA_VERSION",
    "DEFAULT_SCHEDULE",
    "SCHEDULE_DEPTH",
    "assert_detectors_deterministic",
    "generate_bb_memory_circuit",
    "search_schedules",
    "validate_schedule",
]
