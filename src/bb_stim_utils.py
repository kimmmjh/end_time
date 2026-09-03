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

import math
from dataclasses import dataclass, replace
from typing import Sequence

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

# ``legacy`` preserves every circuit/noise location used by schema-v2 results
# produced before the paper profiles were added.  ``standard`` and ``si1000``
# implement Tables II and III of arXiv:2607.05897 on this repository's BB
# R-H-CX-H-M extraction circuit.  The paper studies different tile-code
# schedules; selecting a profile reproduces its channels, not its code layout.
BB_CIRCUIT_NOISE_MODELS = ("legacy", "standard", "si1000")


@dataclass(frozen=True)
class BBCircuitNoiseProfile:
    """Resolved physical error rates for one BB circuit noise model.

    ``base_error_rate`` is the user-facing ``p``.  The remaining rates are
    derived from it for the paper profiles and remain independently
    configurable only for ``legacy``.  The two idle channels are kept
    separate because Table III specifies that they fire independently and
    stack during measurement/reset ticks.
    """

    name: str
    base_error_rate: float
    reset_error_rate: float
    one_qubit_error_rate: float
    two_qubit_error_rate: float
    swap_error_rate: float
    measurement_error_rate: float
    gate_idle_error_rate: float
    resonator_idle_error_rate: float
    full_tick_idle: bool

    def without_noise(self) -> "BBCircuitNoiseProfile":
        """Keep the operation/tick layout while setting every channel to zero."""

        return replace(
            self,
            base_error_rate=0.0,
            reset_error_rate=0.0,
            one_qubit_error_rate=0.0,
            two_qubit_error_rate=0.0,
            swap_error_rate=0.0,
            measurement_error_rate=0.0,
            gate_idle_error_rate=0.0,
            resonator_idle_error_rate=0.0,
        )


def normalize_bb_circuit_noise_model(name: str) -> str:
    """Return a canonical BB circuit profile name, accepting useful aliases."""

    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "legacy": "legacy",
        "custom": "legacy",
        "standard": "standard",
        "paper_standard": "standard",
        "table_ii": "standard",
        "table2": "standard",
        "si1000": "si1000",
        "paper_si1000": "si1000",
        "table_iii": "si1000",
        "table3": "si1000",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "circuit_noise_model must be one of "
            f"{', '.join(BB_CIRCUIT_NOISE_MODELS)}, got {name!r}."
        ) from exc


def _rates_match(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)


def resolve_bb_circuit_noise_profile(
    name: str,
    *,
    base_error_rate: float,
    measurement_error_rate: float | None = None,
    idle_error_rate: float | None = None,
) -> BBCircuitNoiseProfile:
    """Resolve ``legacy``, Table-II standard, or Table-III SI1000 rates.

    The paper models have one free physical parameter.  Optional measurement
    and idle arguments are accepted only when they equal the rate fixed by the
    selected table; this catches commands that look like a paper run while
    silently overriding part of its noise model.
    """

    model = normalize_bb_circuit_noise_model(name)
    p = float(base_error_rate)
    _validate_probability("base_error_rate", p)

    if model == "legacy":
        q = p if measurement_error_rate is None else float(measurement_error_rate)
        idle = 0.0 if idle_error_rate is None else float(idle_error_rate)
        _validate_probability("measurement_error_rate", q)
        _validate_probability("idle_error_rate", idle)
        if p > 0.75:
            raise ValueError(
                "legacy base_error_rate must be at most 0.75 because it uses "
                "DEPOLARIZE1 after Hadamards."
            )
        if idle > 0.75:
            raise ValueError(
                "legacy idle_error_rate must be at most 0.75 for DEPOLARIZE1."
            )
        return BBCircuitNoiseProfile(
            name=model,
            base_error_rate=p,
            reset_error_rate=p,
            one_qubit_error_rate=p,
            two_qubit_error_rate=p,
            swap_error_rate=p,
            measurement_error_rate=q,
            gate_idle_error_rate=idle,
            resonator_idle_error_rate=0.0,
            full_tick_idle=False,
        )

    if model == "standard":
        q = p
        idle = p
        if p > 0.75:
            raise ValueError(
                "standard base_error_rate must be at most 0.75 because Table II "
                "uses DEPOLARIZE1(p) for idle faults."
            )
        one_qubit = 0.0
        reset = p
        swap = p
        resonator = 0.0
    else:
        # Modified SI1000 from Table III.  The measurement channel is the most
        # restrictive probability and requires 5p <= 1.
        if p > 0.2:
            raise ValueError(
                "si1000 base_error_rate must be at most 0.2 because Table III "
                "uses X_ERROR(5p) before measurement."
            )
        q = 5.0 * p
        idle = p / 10.0
        one_qubit = p / 10.0
        reset = 2.0 * p
        swap = 1.5 * p
        resonator = 2.0 * p

    if measurement_error_rate is not None and not _rates_match(
        measurement_error_rate, q
    ):
        raise ValueError(
            f"{model} fixes measurement_error_rate={q:g} from p={p:g}; "
            f"got {float(measurement_error_rate):g}. Use legacy for custom rates."
        )
    if idle_error_rate is not None and not _rates_match(idle_error_rate, idle):
        raise ValueError(
            f"{model} fixes gate idle rate={idle:g} from p={p:g}; "
            f"got {float(idle_error_rate):g}. Use legacy for custom rates."
        )

    return BBCircuitNoiseProfile(
        name=model,
        base_error_rate=p,
        reset_error_rate=reset,
        one_qubit_error_rate=one_qubit,
        two_qubit_error_rate=p,
        swap_error_rate=swap,
        measurement_error_rate=q,
        gate_idle_error_rate=idle,
        resonator_idle_error_rate=resonator,
        full_tick_idle=True,
    )


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


def _append_legacy_cycle(
    circuit: "stim.Circuit",
    *,
    layout: BBCircuitLayout,
    layers: Sequence[Sequence[tuple[int, int]]],
    noise: BBCircuitNoiseProfile,
) -> None:
    """Append the pre-profile circuit exactly, preserving old checkpoints."""

    cells = layout.cells
    x_ancillas = [layout.x_ancilla_base + cell for cell in range(cells)]
    z_ancillas = [layout.z_ancilla_base + cell for cell in range(cells)]
    ancillas = x_ancillas + z_ancillas
    data = list(range(2 * cells))
    noisy = noise.base_error_rate > 0.0

    circuit.append("R", ancillas)
    if noisy:
        circuit.append("X_ERROR", ancillas, noise.reset_error_rate)
    circuit.append("H", x_ancillas)
    if noisy:
        circuit.append("DEPOLARIZE1", x_ancillas, noise.one_qubit_error_rate)

    for layer in layers:
        targets = [qubit for pair in layer for qubit in pair]
        circuit.append("CX", targets)
        if noisy:
            circuit.append("DEPOLARIZE2", targets, noise.two_qubit_error_rate)
        if noise.gate_idle_error_rate > 0.0:
            busy = set(targets)
            idle = [qubit for qubit in data if qubit not in busy]
            if idle:
                circuit.append(
                    "DEPOLARIZE1", idle, noise.gate_idle_error_rate
                )

    circuit.append("H", x_ancillas)
    if noisy:
        circuit.append("DEPOLARIZE1", x_ancillas, noise.one_qubit_error_rate)
    if noise.measurement_error_rate > 0.0:
        circuit.append("X_ERROR", ancillas, noise.measurement_error_rate)
    circuit.append("M", ancillas)


def _append_idle_noise(
    circuit: "stim.Circuit",
    targets: Sequence[int],
    noise: BBCircuitNoiseProfile,
    *,
    measurement_or_reset_tick: bool = False,
) -> None:
    """Apply the paper's independent idle channels to inactive qubits."""

    if not targets:
        return
    if noise.gate_idle_error_rate > 0.0:
        circuit.append("DEPOLARIZE1", targets, noise.gate_idle_error_rate)
    if measurement_or_reset_tick and noise.resonator_idle_error_rate > 0.0:
        # Table III states that resonator idle stacks independently with gate
        # idle on measurement/reset ticks, so do not combine the probabilities.
        circuit.append(
            "DEPOLARIZE1", targets, noise.resonator_idle_error_rate
        )


def _append_paper_cycle(
    circuit: "stim.Circuit",
    *,
    layout: BBCircuitLayout,
    layers: Sequence[Sequence[tuple[int, int]]],
    noise: BBCircuitNoiseProfile,
) -> None:
    """Append one tick-aware Table-II/Table-III BB extraction cycle.

    X-basis preparation/measurement is expressed using the repository's
    physical ``R-H`` and ``H-M`` sequence.  Therefore an X flip after ``R`` is
    equivalent to Table II's Z flip after ``InitX``, and an X flip after the
    final H is equivalent to its Z flip before ``MeasX``.  Table II defines no
    active 1Q-gate channel, while Table III applies DEP1(p/10) after each H.
    """

    cells = layout.cells
    x_ancillas = [layout.x_ancilla_base + cell for cell in range(cells)]
    z_ancillas = [layout.z_ancilla_base + cell for cell in range(cells)]
    ancillas = x_ancillas + z_ancillas
    data = list(range(2 * cells))
    all_qubits = data + ancillas

    # Reset tick.  Data qubits are inactive; under SI1000 they receive both
    # gate-idle and resonator-idle noise independently.
    circuit.append("R", ancillas)
    if noise.reset_error_rate > 0.0:
        circuit.append("X_ERROR", ancillas, noise.reset_error_rate)
    _append_idle_noise(
        circuit, data, noise, measurement_or_reset_tick=True
    )
    circuit.append("TICK")

    # X-basis preparation tick.
    circuit.append("H", x_ancillas)
    if noise.one_qubit_error_rate > 0.0:
        circuit.append("DEPOLARIZE1", x_ancillas, noise.one_qubit_error_rate)
    _append_idle_noise(circuit, data + z_ancillas, noise)
    circuit.append("TICK")

    for layer in layers:
        targets = [qubit for pair in layer for qubit in pair]
        circuit.append("CX", targets)
        if noise.two_qubit_error_rate > 0.0:
            circuit.append("DEPOLARIZE2", targets, noise.two_qubit_error_rate)
        busy = set(targets)
        _append_idle_noise(
            circuit, [qubit for qubit in all_qubits if qubit not in busy], noise
        )
        circuit.append("TICK")

    # X-basis measurement rotation tick.
    circuit.append("H", x_ancillas)
    if noise.one_qubit_error_rate > 0.0:
        circuit.append("DEPOLARIZE1", x_ancillas, noise.one_qubit_error_rate)
    _append_idle_noise(circuit, data + z_ancillas, noise)
    circuit.append("TICK")

    # Measurement tick.  The pre-measurement X channel is a classical readout
    # flip in the measured basis after the X ancillas' final H.
    if noise.measurement_error_rate > 0.0:
        circuit.append("X_ERROR", ancillas, noise.measurement_error_rate)
    circuit.append("M", ancillas)
    _append_idle_noise(
        circuit, data, noise, measurement_or_reset_tick=True
    )
    circuit.append("TICK")


def _append_cycle(
    circuit: "stim.Circuit",
    *,
    layout: BBCircuitLayout,
    layers: Sequence[Sequence[tuple[int, int]]],
    noise: BBCircuitNoiseProfile,
) -> None:
    """Append one cycle using the selected circuit noise profile."""

    if noise.name == "legacy":
        _append_legacy_cycle(circuit, layout=layout, layers=layers, noise=noise)
    else:
        _append_paper_cycle(circuit, layout=layout, layers=layers, noise=noise)


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
    measurement_error_rate: float | None = None,
    idle_error_rate: float | None = None,
    circuit_noise_model: str = "legacy",
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

    ``circuit_noise_model='legacy'`` preserves the original toric-style
    convention: a bit flip after reset, one-qubit depolarizing after each
    Hadamard, two-qubit depolarizing after each CNOT and a configurable readout
    flip, with optional data-only CNOT-layer idle noise.

    ``standard`` and ``si1000`` implement Tables II and III of
    arXiv:2607.05897 on this seven-layer periodic-BB schedule. Both track every
    inactive data and ancilla qubit on every physical tick. Standard uses p
    for preparation, CNOT, measurement and idle channels and has ideal H
    gates. Modified SI1000 uses reset=2p, H/gate-idle=p/10, CNOT=p,
    measurement=5p and an additional resonator-idle=2p on measurement/reset
    ticks. The reference and closing cycles remain perfect under every
    profile; the selected model controls only the requested noisy cycles.

    Set ``check_schedule=False`` only to explore alternative CNOT orderings.
    The layer-collision check still runs, but the determinism condition is
    skipped, so the caller must verify the result with
    :func:`assert_detectors_deterministic` before trusting any decoding result.
    """

    _require_stim()
    if isinstance(code, str):
        code = BBCodeSpec.from_name(code)
    noise = resolve_bb_circuit_noise_profile(
        circuit_noise_model,
        base_error_rate=gate_error_rate,
        measurement_error_rate=measurement_error_rate,
        idle_error_rate=idle_error_rate,
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
        noise=noise.without_noise(),
    )
    _append_logical_sheets(circuit, code)

    for time_index in range(rounds):
        _append_cycle(
            circuit,
            layout=layout,
            layers=layers,
            noise=noise,
        )
        _append_detector_frame(circuit, layout=layout, time_index=time_index)

    # This is a boundary measurement, not one of the requested noisy rounds.
    # Keeping it separate preserves every final-cycle hook/data fault in the
    # detector error model.
    _append_cycle(
        circuit,
        layout=layout,
        layers=layers,
        noise=noise.without_noise(),
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
    "BB_CIRCUIT_NOISE_MODELS",
    "BBCircuitLayout",
    "BBCircuitNoiseProfile",
    "CIRCUIT_SCHEMA_VERSION",
    "DEFAULT_SCHEDULE",
    "SCHEDULE_DEPTH",
    "assert_detectors_deterministic",
    "generate_bb_memory_circuit",
    "normalize_bb_circuit_noise_model",
    "resolve_bb_circuit_noise_profile",
    "search_schedules",
    "validate_schedule",
]
