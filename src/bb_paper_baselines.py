"""BB memory circuits and library BP/OSD baselines, independent of neural BP.

The eight-tick extraction schedule follows the authors' public implementation:
https://github.com/sbravyi/BivariateBicycleCodes/blob/
fa77e3333d3ec44c79d8f914dd24c040d1da471b/decoder_setup.py
The circuit uses native X/Z preparation and measurement (no inserted Hadamard
ticks), noisy idle locations, and the logical-X memory experiment of Blue et al.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import stim
from ldpc import BpOsdDecoder

from .bb_code import BBCodeSpec

CIRCUIT_POLICY = "bravyi_8tick_plus_memory_v1"
SOURCE_COMMIT = "fa77e3333d3ec44c79d8f914dd24c040d1da471b"
X_ORDER = (None, 1, 4, 3, 5, 0, 2)
Z_ORDER = (3, 5, 0, 1, 2, 4, None)
METHODS = ("bp", "bposd0", "bposd_cs3")


def cnot_layers(code):
    """Return the paper's seven CNOT layers in this repository's qubit order."""
    cells = code.cells
    a = ((3, 0), (0, 1), (0, 2))
    b = ((0, 3), (1, 0), (2, 0))

    def shifted(cell, displacement):
        i, j = divmod(cell, code.m)
        dx, dy = displacement
        return ((i + dx) % code.ell) * code.m + (j + dy) % code.m

    layers = []
    hx, hz = np.zeros_like(code.hx), np.zeros_like(code.hz)
    for x_direction, z_direction in zip(X_ORDER, Z_ORDER):
        layer = []
        for cell in range(cells):
            if x_direction is not None:
                block, term = divmod(x_direction, 3)
                data = block * cells + shifted(cell, (a, b)[block][term])
                layer.append((code.n + cell, data))
                hx[cell, data] ^= 1
            if z_direction is not None:
                block, term = divmod(z_direction, 3)
                dx, dy = (b, a)[block][term]
                data = block * cells + shifted(cell, (-dx, -dy))
                layer.append((data, code.n + cells + cell))
                hz[cell, data] ^= 1
        flat = [q for pair in layer for q in pair]
        if len(flat) != len(set(flat)):
            raise ValueError("Paper schedule has simultaneous qubit collisions.")
        layers.append(layer)
    if not np.array_equal(hx, code.hx) or not np.array_equal(hz, code.hz):
        raise ValueError("Scheduled gates disagree with the BB check matrices.")
    return layers


def make_memory_circuit(code, *, p, rounds=None):
    """Ideal |+> input, reference, d noisy cycles, perfect closing, logical X.

    Noise per location: preparation/measurement flip p, idle DEP1(p), and
    CNOT DEP2(p). Native RX/MX implement the paper's InitX/MeasX operations.
    Detectors compare successive rounds; their coordinates start with check
    type (0=X, 1=Z). All k observables are logical X measurement flips.
    """
    code = BBCodeSpec.from_name(code) if isinstance(code, str) else code
    rounds = code.d if rounds is None else rounds
    if not isinstance(rounds, int) or rounds < 1 or not 0 <= p < .5:
        raise ValueError("Require integer rounds >= 1 and 0 <= p < .5.")
    cells = code.cells
    data = list(range(code.n))
    xs = list(range(code.n, code.n + cells))
    zs = list(range(code.n + cells, 2 * code.n))
    layers = cnot_layers(code)
    circuit = stim.Circuit()
    circuit.append("RX", data)
    circuit.append("R", zs)

    def cycle(rate):
        for tick in range(8):
            busy = set()
            if tick == 0:
                circuit.append("RX", xs)
                if rate:
                    circuit.append("Z_ERROR", xs, rate)
                busy.update(xs)
            if tick == 6:
                if rate:
                    circuit.append("X_ERROR", zs, rate)
                circuit.append("M", zs)
                busy.update(zs)
            if tick == 7:
                if rate:
                    circuit.append("Z_ERROR", xs, rate)
                circuit.append("MX", xs)
                circuit.append("R", zs)
                if rate:
                    circuit.append("X_ERROR", zs, rate)
                busy.update(xs + zs)
            else:
                targets = [q for pair in layers[tick] for q in pair]
                circuit.append("CX", targets)
                if rate:
                    circuit.append("DEPOLARIZE2", targets, rate)
                busy.update(targets)
            idle = [q for q in range(2 * code.n) if q not in busy]
            if rate and idle:
                circuit.append("DEPOLARIZE1", idle, rate)
            circuit.append("TICK")

    cycle(0.)
    for frame in range(rounds + 1):
        cycle(p if frame < rounds else 0.)
        # Each cycle measures Z checks first and X checks second.
        for kind, offset in ((0, -cells), (1, -2 * cells)):
            for cell in range(cells):
                circuit.append("DETECTOR", [stim.target_rec(offset + cell),
                    stim.target_rec(offset + cell - 2 * cells)], [kind, cell, frame])
    circuit.append("MX", data)
    for index, logical in enumerate(code.logicals_x):
        circuit.append("OBSERVABLE_INCLUDE",
                       [stim.target_rec(-code.n + int(q)) for q in np.flatnonzero(logical)],
                       index)
    # Stim independently verifies stabilizer propagation and the logical boundary.
    circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    return circuit


@dataclass
class DecodingProblem:
    check: sp.csr_matrix
    logical: sp.csr_matrix
    priors: np.ndarray
    detector_indices: np.ndarray


def decoding_problem(circuit, detector_input):
    """Project a DEM to X checks or joint X/Z checks, then merge equal columns.

    X means the measured X checks (detecting Z faults), not X-error decoding.
    XZ means a single joint DEM, not two independently decoded CSS sectors.
    """
    if detector_input not in ("x", "xz"):
        raise ValueError("detector_input must be x or xz")
    coordinates = circuit.get_detector_coordinates()
    selected = np.array([i for i in range(circuit.num_detectors)
                         if detector_input == "xz" or coordinates[i][0] == 0], dtype=np.int64)
    reindex = {int(old): new for new, old in enumerate(selected)}
    columns = {}
    dem = circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        ds, ls = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id() and target.val in reindex:
                ds.symmetric_difference_update((reindex[target.val],))
            elif target.is_logical_observable_id():
                ls.symmetric_difference_update((target.val,))
        if not ds:
            if ls:
                raise ValueError("Undetectable logical mechanism after DEM projection.")
            continue
        key, action = tuple(sorted(ds)), tuple(sorted(ls))
        rate = instruction.args_copy()[0]
        if key in columns:
            previous_action, previous_rate = columns[key]
            if action != previous_action:
                raise ValueError("Identical detector columns have different logical actions.")
            rate = previous_rate * (1 - rate) + (1 - previous_rate) * rate
        columns[key] = (action, rate)
    if not columns:
        raise ValueError("No noisy mechanisms in the decoding problem.")
    dr, dc, lr, lc, priors = [], [], [], [], []
    for col, (detectors, (logicals, rate)) in enumerate(columns.items()):
        dr.extend(detectors); dc.extend([col] * len(detectors))
        lr.extend(logicals); lc.extend([col] * len(logicals))
        priors.append(rate)
    check = sp.csr_matrix((np.ones(len(dr), dtype=np.uint8), (dr, dc)),
                          shape=(len(selected), len(columns)))
    logical = sp.csr_matrix((np.ones(len(lr), dtype=np.uint8), (lr, lc)),
                            shape=(circuit.num_observables, len(columns)))
    if np.any(np.diff(check.indptr) == 0):
        raise ValueError("Selected detector has no noisy mechanism.")
    return DecodingProblem(check, logical, np.array(priors), selected)


class PairedBpOsd:
    """One library BP trajectory gives plain BP, BP+OSD-0 and BP+CS3 outputs."""

    def __init__(self, problem, max_iterations=1000):
        self.problem = problem
        self.decoder = BpOsdDecoder(
            problem.check, error_channel=problem.priors.tolist(),
            bp_method="ms", ms_scaling_factor=1., max_iter=max_iterations,
            schedule="parallel", omp_thread_count=1,
            osd_method="OSD_CS", osd_order=3,
        )

    def decode(self, syndrome):
        if not np.any(syndrome):
            zero = np.zeros(self.problem.check.shape[1], dtype=np.uint8)
            return (zero, zero, zero), True, 0
        best = self.decoder.decode(np.asarray(syndrome, dtype=np.uint8)).copy()
        bp = self.decoder.bp_decoding.copy()
        converged = bool(self.decoder.converge)
        osd0 = bp.copy() if converged else self.decoder.osd0_decoding.copy()
        return (bp, osd0, best), converged, int(self.decoder.iter)

    def score(self, detectors, observables):
        n = len(detectors)
        result = {method: {
            "success": np.empty(n, dtype=bool), "converged": np.empty(n, dtype=bool),
            "logical_mismatch": np.empty(n, dtype=bool),
            "prediction": np.empty_like(observables, dtype=np.uint8),
            "correction_packed": np.empty((n, (self.problem.check.shape[1] + 7) // 8),
                                           dtype=np.uint8),
        } for method in METHODS}
        iterations = np.empty(n, dtype=np.int32)
        for i, shot in enumerate(detectors):
            corrections, bp_converged, iterations[i] = self.decode(shot)
            for method, correction in zip(METHODS, corrections):
                valid = np.array_equal((self.problem.check @ correction) % 2, shot)
                prediction = (self.problem.logical @ correction) % 2
                mismatch = bool(np.any(prediction != observables[i]))
                if method == "bp" and valid != bp_converged:
                    raise RuntimeError("Library BP flag disagrees with the syndrome check.")
                if method != "bp" and not valid:
                    raise RuntimeError("OSD returned a syndrome-invalid correction.")
                result[method]["success"][i] = valid and not mismatch
                result[method]["converged"][i] = valid
                result[method]["logical_mismatch"][i] = mismatch
                result[method]["prediction"][i] = prediction
                result[method]["correction_packed"][i] = np.packbits(correction, bitorder="little")
        return result, iterations
