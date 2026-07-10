"""Stim circuits for circuit-level toric-code data generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

try:
    import stim
except ImportError:  # Keep non-circuit noise modes usable without Stim.
    stim = None

if TYPE_CHECKING:
    from panqec.codes import StabilizerCode


# This order is a valid four-colouring of the toric-code interaction graph and
# makes consecutive stabilizer measurements quantum non-demolition. In
# particular, swapping the last two directions makes some detectors random
# even in a noiseless circuit.
_CNOT_DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require_stim() -> None:
    if stim is None:
        raise ImportError(
            "Circuit-level noise requires Stim. Install the project "
            "dependencies with `pip install -r requirements.txt`."
        )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _ordered_stabilizers(code: StabilizerCode) -> list[dict]:
    stabilizers: list[dict | None] = [None] * len(code.stabilizer_index)
    for location, index in code.stabilizer_index.items():
        stabilizers[index] = code.get_stabilizer(location)

    if any(operator is None for operator in stabilizers):
        raise ValueError("PanQEC stabilizer indices must be contiguous.")
    return stabilizers  # type: ignore[return-value]


def _interaction_layers(
    code: StabilizerCode,
) -> list[list[tuple[int, int, str]]]:
    """Return four collision-free (stabilizer, data, Pauli) CNOT layers."""
    by_direction = {direction: [] for direction in _CNOT_DIRECTIONS}
    period = 2 * np.asarray(code.size, dtype=int)

    for stabilizer_location, stabilizer_index in code.stabilizer_index.items():
        for qubit_location, pauli in code.get_stabilizer(
            stabilizer_location
        ).items():
            delta = np.asarray(qubit_location) - np.asarray(stabilizer_location)
            delta = np.where(delta > period // 2, delta - period, delta)
            delta = np.where(delta < -(period // 2), delta + period, delta)
            direction = tuple(int(component) for component in delta)
            if direction not in by_direction:
                raise ValueError(
                    "Only nearest-neighbour square-lattice stabilizers are "
                    f"supported; found displacement {direction}."
                )
            by_direction[direction].append(
                (stabilizer_index, code.qubit_index[qubit_location], pauli)
            )

    layers = [by_direction[direction] for direction in _CNOT_DIRECTIONS]
    num_data = code.n
    for layer in layers:
        targets = [
            target
            for stabilizer_index, data_index, _ in layer
            for target in (num_data + stabilizer_index, data_index)
        ]
        if len(targets) != len(set(targets)):
            raise ValueError("The generated CNOT schedule contains a collision.")
    return layers


def _append_logical_sheets(
    circuit: stim.Circuit,
    code: StabilizerCode,
) -> None:
    """Mark Pauli-frame correlation sheets in [logical X, logical Z] order."""
    target_for_pauli = {
        "X": stim.target_x,
        "Y": stim.target_y,
        "Z": stim.target_z,
    }
    logicals = [*code.get_logicals_x(), *code.get_logicals_z()]
    for observable_index, operator in enumerate(logicals):
        targets = [
            target_for_pauli[pauli](code.qubit_index[location])
            for location, pauli in operator.items()
        ]
        circuit.append("OBSERVABLE_INCLUDE", targets, observable_index)


def _append_measurement_cycle(
    circuit: stim.Circuit,
    *,
    code: StabilizerCode,
    x_stabilizers: Iterable[int],
    layers: list[list[tuple[int, int, str]]],
    gate_error_rate: float,
    measurement_error_rate: float,
) -> None:
    """Append one ancilla-based stabilizer measurement cycle."""
    num_data = code.n
    num_stabilizers = len(code.stabilizer_index)
    ancillas = [num_data + index for index in range(num_stabilizers)]
    x_ancillas = [num_data + index for index in x_stabilizers]
    noisy = gate_error_rate > 0

    circuit.append("R", ancillas)
    if noisy:
        # A bit flip after reset models preparation failure.
        circuit.append("X_ERROR", ancillas, gate_error_rate)

    circuit.append("H", x_ancillas)
    if noisy:
        circuit.append("DEPOLARIZE1", x_ancillas, gate_error_rate)

    for layer in layers:
        targets = []
        for stabilizer_index, data_index, pauli in layer:
            ancilla_index = num_data + stabilizer_index
            if pauli == "X":
                targets.extend((ancilla_index, data_index))
            elif pauli == "Z":
                targets.extend((data_index, ancilla_index))
            else:
                raise ValueError(
                    "Circuit-level generation currently supports CSS "
                    f"stabilizers only, got {pauli!r}."
                )
        circuit.append("CX", targets)
        if noisy:
            circuit.append("DEPOLARIZE2", targets, gate_error_rate)

    circuit.append("H", x_ancillas)
    if noisy:
        circuit.append("DEPOLARIZE1", x_ancillas, gate_error_rate)
    if measurement_error_rate > 0:
        circuit.append("X_ERROR", ancillas, measurement_error_rate)
    circuit.append("M", ancillas)


def generate_toric_memory_circuit(
    code: StabilizerCode,
    *,
    rounds: int,
    gate_error_rate: float,
    measurement_error_rate: float,
) -> stim.Circuit:
    """Build a noisy toric-code memory circuit with detector annotations.

    The returned detector order is ``(time, check_type, x, y)`` with all
    vertex checks followed by all face checks at each time. Logical
    observables are correlation sheets in PanQEC's
    ``[logicals_x, logicals_z]`` order, so X, Y, and Z logical components
    from the same physical shot remain correlated.
    """
    _require_stim()
    _validate_probability("gate_error_rate", gate_error_rate)
    _validate_probability("measurement_error_rate", measurement_error_rate)
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}.")
    if len(code.size) != 2 or code.size[0] != code.size[1]:
        raise ValueError("The current model requires a square 2D code.")

    lattice_size = code.size[0]
    num_data = code.n
    num_stabilizers = len(code.stabilizer_index)
    expected_stabilizers = 2 * lattice_size**2
    if num_stabilizers != expected_stabilizers:
        raise ValueError(
            "Expected 2*L^2 toric-code stabilizers, got "
            f"{num_stabilizers} for L={lattice_size}."
        )

    stabilizers = _ordered_stabilizers(code)
    x_stabilizers = [
        index
        for index, operator in enumerate(stabilizers)
        if set(operator.values()) == {"X"}
    ]
    z_stabilizers = [
        index
        for index, operator in enumerate(stabilizers)
        if set(operator.values()) == {"Z"}
    ]
    if len(x_stabilizers) != lattice_size**2 or len(z_stabilizers) != lattice_size**2:
        raise ValueError("Expected L^2 pure-X checks and L^2 pure-Z checks.")

    layers = _interaction_layers(code)
    z_stabilizer_set = set(z_stabilizers)
    circuit = stim.Circuit()
    circuit.append("R", range(num_data))

    # Establish a noiseless reference syndrome. This projects the initially
    # unencoded product state into a stabilizer eigenspace without requiring
    # first-round dummy detectors.
    _append_measurement_cycle(
        circuit,
        code=code,
        x_stabilizers=x_stabilizers,
        layers=layers,
        gate_error_rate=0.0,
        measurement_error_rate=0.0,
    )
    _append_logical_sheets(circuit, code)

    for time_index in range(rounds):
        _append_measurement_cycle(
            circuit,
            code=code,
            x_stabilizers=x_stabilizers,
            layers=layers,
            gate_error_rate=gate_error_rate,
            measurement_error_rate=measurement_error_rate,
        )
        for stabilizer_index, location in enumerate(code.stabilizer_coordinates):
            check_type = 0 if stabilizer_index in z_stabilizer_set else 1
            x, y = (coordinate // 2 for coordinate in location)
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-num_stabilizers + stabilizer_index),
                    stim.target_rec(-2 * num_stabilizers + stabilizer_index),
                ],
                [float(check_type), float(x), float(y), float(time_index)],
            )

    _append_logical_sheets(circuit, code)
    return circuit
